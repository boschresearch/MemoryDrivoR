# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .score_module.scorer import Scorer
from .score_module.score_aggregation import (
    aggregate_multiplicative_weighted_score,
    apply_multiplicative_metric_weight,
)
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from .epi_mem_modules import EpiMemoryInjector, EpisodicMemoryBank
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)

class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.embed_dims = self._config.tf_d_model
        self.b2d = bool(config.get("b2d", False))
        self.use_acceleration = bool(config.get("use_acceleration", False))
        self.use_local_command_xy = bool(config.get("use_local_command_xy", True))

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1

        # create the image backbone
        if self.num_cams > 0:
            config_image_backbone = config["image_backbone"]
            config_image_backbone["image_size"] = config["image_size"]
            config_image_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_image_backbone["tf_d_model"] = config["tf_d_model"]
            self.image_backbone = ImgEncoder(config_image_backbone)
            self.scene_embeds = nn.Parameter(torch.randn(1, self.num_cams, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

            # print("self.scene_embeds ", self.scene_embeds)

        # create the lidar backbone
        if self.num_lidar > 0:
            config_lidar_backbone = config["lidar_backbone"]
            config_lidar_backbone["image_size"] = config["lidar_image_size"]
            config_lidar_backbone["num_scene_tokens"] = config["num_scene_tokens"]
            config_lidar_backbone["tf_d_model"] = config["tf_d_model"]
            self.lidar_backbone = ImgEncoder(config_lidar_backbone)
            self.lidar_scene_embeds = nn.Parameter(torch.randn(1, self.num_lidar, self._config.num_scene_tokens, self.image_backbone.num_features)*1e-6, requires_grad=True)

        # ego status encoder
        ego_status_dim = self.get_ego_status_feature_dim()
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(ego_status_dim * 4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(ego_status_dim, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(
            num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(self.poses_num * 3, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        # get the trajectory decoders
        self.poses_num = config.num_poses
        self.state_size = 3
        ref_num = config.ref_num
        self.traj_head = nn.ModuleList(
            [MLP(config.tf_d_model, config.tf_d_ffn, traj_head_output_size) for _ in range(ref_num + 1)]
        )

        # scorer
        self.scorer = Scorer(config)

        # episodic memory
        self.epi_cfg = config["epi_memory"] if "epi_memory" in config else None
        self.use_episodic_memory = bool(self.epi_cfg is not None and self.epi_cfg.get("enabled", False))
        self.log_no_memories_in_forward = bool(
            self.epi_cfg is not None and self.epi_cfg.get("log_no_memories_in_forward", False)
        )
        self.epi_memory_injector = None
        self.epi_memory_bank = None
        self.freeze_backbone = bool(config.get("freeze_backbone", False))
        self.skip_perception_backbone = bool(config.get("skip_perception_backbone", False))
        if self.use_episodic_memory:
            injector_cfg = (
                OmegaConf.to_container(self.epi_cfg.injector, resolve=True)
                if "injector" in self.epi_cfg and self.epi_cfg.injector is not None
                else {}
            )
            injector_cfg["dim"] = config.tf_d_model
            injector_cfg.setdefault("out_dim", config.tf_d_model)
            self.epi_memory_injector = EpiMemoryInjector(**injector_cfg)

            bank_cfg = (
                OmegaConf.to_container(self.epi_cfg.bank, resolve=True)
                if "bank" in self.epi_cfg and self.epi_cfg.bank is not None
                else {}
            )
            self.epi_memory_bank = EpisodicMemoryBank(
                bank_path=bank_cfg.get("path", None),
                top_k=bank_cfg.get("top_k", 0),
                use_precomputed_train_neighbors=bank_cfg.get("use_precomputed_train_neighbors", True),
                yaw_distance_weight=bank_cfg.get("yaw_distance_weight", 0.0),
                exclude_self=bank_cfg.get("exclude_self", True),
                max_distance_m=bank_cfg.get("max_distance_m", None),
                min_yaw=bank_cfg.get("min_yaw", None),
                max_yaw=bank_cfg.get("max_yaw", None),
            )

        if self.freeze_backbone:
            self._freeze_backbones()

    def get_ego_status_feature_dim(self) -> int:
        if self.b2d:
            ego_status_dim = 7
            if self.use_acceleration:
                ego_status_dim += 2
            if self.use_local_command_xy:
                ego_status_dim += 2
            return ego_status_dim
        return 11

    def get_b2d_ego_status_indices(self) -> list[int]:
        indices = [0]
        if self.use_acceleration:
            indices.extend([1, 2])
        if self.use_local_command_xy:
            indices.extend([3, 4])
        indices.extend([5, 6, 7, 8, 9, 10])
        return indices

    def select_b2d_ego_status_features(self, ego_status: torch.Tensor) -> torch.Tensor:
        if ego_status.shape[-1] != 11:
            raise ValueError(f"Unexpected B2D ego_status dim {ego_status.shape[-1]}; expected 11.")
        return ego_status[..., self.get_b2d_ego_status_indices()]

    def _freeze_backbones(self) -> None:
        if hasattr(self, "image_backbone"):
            for param in self.image_backbone.parameters():
                param.requires_grad = False
            self.image_backbone.eval()
            if hasattr(self.image_backbone, "use_grid_mask"):
                self.image_backbone.use_grid_mask = False
            self.scene_embeds.requires_grad_(False)

        if hasattr(self, "lidar_backbone"):
            for param in self.lidar_backbone.parameters():
                param.requires_grad = False
            self.lidar_backbone.eval()
            if hasattr(self.lidar_backbone, "use_grid_mask"):
                self.lidar_backbone.use_grid_mask = False
            self.lidar_scene_embeds.requires_grad_(False)

    def encode_scene_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = features["ego_status"].shape[0]
        scene_features = []

        # image features
        if self.num_cams > 0:
            if "image" in features:
                img = features["image"]
            elif "camera_feature" in features:
                img = features["camera_feature"]
            else:
                raise ValueError("Missing image input in features.")

            scene_tokens = self.scene_embeds.repeat(batch_size, 1, 1, 1)
            image_ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
            with image_ctx:
                image_scene_tokens = self.image_backbone(img, scene_tokens)

            log.debug(f"Backbone image - {image_scene_tokens.shape}")
            scene_features.append(image_scene_tokens)

        # lidar features
        if self.num_lidar > 0:
            img = features["lidar_feature"]
            scene_tokens = self.lidar_scene_embeds.repeat(batch_size, 1, 1, 1)
            lidar_ctx = torch.no_grad() if self.freeze_backbone else nullcontext()
            with lidar_ctx:
                lidar_scene_tokens = self.lidar_backbone(img, scene_tokens)
            log.debug(f"Backbone lidar - {lidar_scene_tokens.shape}")
            scene_features.append(lidar_scene_tokens)

        scene_features = torch.cat(scene_features, dim=1)
        log.debug(f"Scene features - {scene_features.shape}")
        return scene_features

    def _inject_episodic_memory(
        self,
        scene_features: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if not self.use_episodic_memory or self.epi_memory_injector is None or self.epi_memory_bank is None:
            return scene_features, None, {}

        if self.epi_memory_bank.top_k <= 0:
            return scene_features, None, {}

        if "ego_global_pose" not in features:
            log.warning("Episodic memory enabled but `ego_global_pose` is missing from features.")
            return scene_features, None, {}

        query_pose = features["ego_global_pose"]
        scenario_tokens = features.get("scenario_token", None)
        scenario_log_names = features.get("scenario_log_name", None)
        scenario_town_names = features.get("scenario_town_name", None)
        epi_tokens, epi_poses = self.epi_memory_bank.query(
            query_global_pose=query_pose,
            scenario_tokens=scenario_tokens,
            scenario_log_names=scenario_log_names,
            scenario_town_names=scenario_town_names,
            device=scene_features.device,
        )
        self._log_no_memories_in_forward_pass(
            epi_tokens,
            query_pose,
            scenario_tokens,
            scenario_log_names,
            scenario_town_names,
        )
        epi_metrics = self._build_epi_retrieval_metrics(epi_tokens, device=scene_features.device)
        injected_scene_features = self.epi_memory_injector(scene_features, epi_tokens, epi_poses)

        if isinstance(injected_scene_features, torch.Tensor):
            return injected_scene_features, None, epi_metrics

        # Padding for variable-length memories.
        batch_size = len(injected_scene_features)
        max_tokens = max(feat.shape[0] for feat in injected_scene_features)
        padded_features = []
        scene_padding_mask = torch.ones(
            (batch_size, max_tokens), dtype=torch.bool, device=scene_features.device
        )
        for sample_idx, feat in enumerate(injected_scene_features):
            valid_tokens = feat.shape[0]
            if feat.shape[0] < max_tokens:
                pad = feat.new_zeros((max_tokens - feat.shape[0], feat.shape[1]))
                feat = torch.cat([feat, pad], dim=0)
            padded_features.append(feat)
            scene_padding_mask[sample_idx, :valid_tokens] = False
        return torch.stack(padded_features, dim=0), scene_padding_mask, epi_metrics

    @staticmethod
    def _build_epi_retrieval_metrics(
        epi_tokens: List[torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if len(epi_tokens) == 0:
            return {}

        retrieved_counts = torch.tensor(
            [int(mem.shape[0]) for mem in epi_tokens],
            dtype=torch.float32,
            device=device,
        )
        return {
            "retrieval_avg_count": retrieved_counts.mean(),
            "retrieval_zero_frac": (retrieved_counts == 0).float().mean(),
        }

    @staticmethod
    def _get_optional_batch_log_value(values: object, batch_idx: int) -> Optional[str]:
        if values is None:
            return None
        if isinstance(values, str):
            return values if batch_idx == 0 else None
        if isinstance(values, torch.Tensor):
            if values.ndim == 0:
                return str(values.item()) if batch_idx == 0 else None
            if batch_idx >= values.shape[0]:
                return None
            value = values[batch_idx]
            return str(value.item()) if value.numel() == 1 else str(value.detach().cpu().tolist())
        try:
            if batch_idx >= len(values):
                return None
            value = values[batch_idx]
        except TypeError:
            return str(values) if batch_idx == 0 else None
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return str(value.item()) if value.numel() == 1 else str(value.detach().cpu().tolist())
        return str(value)

    def _log_no_memories_in_forward_pass(
        self,
        epi_tokens: List[torch.Tensor],
        query_pose: torch.Tensor,
        scenario_tokens: object,
        scenario_log_names: object,
        scenario_town_names: object,
    ) -> None:
        if not self.log_no_memories_in_forward:
            return

        zero_indices = [batch_idx for batch_idx, mem in enumerate(epi_tokens) if int(mem.shape[0]) == 0]
        if len(zero_indices) == 0:
            return

        query_pose_for_log = query_pose[:, -1] if query_pose.ndim == 3 else query_pose
        for batch_idx in zero_indices:
            pose_xyh = query_pose_for_log[batch_idx, :3].detach().cpu().tolist()
            scenario_token = self._get_optional_batch_log_value(scenario_tokens, batch_idx) or "<unknown>"
            scenario_log_name = self._get_optional_batch_log_value(scenario_log_names, batch_idx) or "<unknown>"
            scenario_town_name = self._get_optional_batch_log_value(scenario_town_names, batch_idx) or "<unknown>"
            log.warning(
                "Episodic retrieval returned no memories for batch_idx=%d token=%s log_name=%s town_name=%s "
                "query_pose=(%.3f, %.3f, %.3f)",
                batch_idx,
                scenario_token,
                scenario_log_name,
                scenario_town_name,
                pose_xyh[0],
                pose_xyh[1],
                pose_xyh[2],
            )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ego status and initial traj tokens
        ego_status_all = features["ego_status"]
        if self.b2d:
            ego_status_all = self.select_b2d_ego_status_features(ego_status_all)
        if self._config.full_history_status:
            ego_status: torch.Tensor = ego_status_all.flatten(-2)
        else:
            ego_status: torch.Tensor = ego_status_all[:, -1]

        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")

        batch_size = ego_status.shape[0]
        if self.skip_perception_backbone:
            scene_features = ego_token.new_zeros((batch_size, 0, self.embed_dims))
        else:
            scene_features = self.encode_scene_features(features)
        scene_features, scene_padding_mask, epi_metrics = self._inject_episodic_memory(scene_features, features)

        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode trajectories
        token_list = self.trajectory_decoder(
            traj_tokens,
            scene_features,
            x_cross_padding_mask=scene_padding_mask,
        )
        log.debug(f"Trajectory decoder - {len(token_list)}")
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i + 1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)

        proposals = proposal_list[-1]
        

        output = {}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        # scoring
        B, N, _, _ = proposals.shape
        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())
        tr_out = self.scorer_attention(
            embedded_traj,
            scene_features,
            x_cross_padding_mask=scene_padding_mask,
        )
        tr_out = tr_out + ego_token
        pred_logit, pred_logit2, pred_agents_states, pred_area_logit, bev_semantic_map, agent_states, agent_labels = self.scorer(
            proposals, tr_out
        )

        output["pred_logit"] = pred_logit
        output["pred_logit2"] = pred_logit2
        output["pred_agents_states"] = pred_agents_states
        output["pred_area_logit"] = pred_area_logit
        output["bev_semantic_map"] = bev_semantic_map
        output["agent_states"] = agent_states
        output["agent_labels"] = agent_labels
        output["epi_metrics"] = epi_metrics

        if self.b2d:
            no_collision_score = pred_logit["no_at_fault_collisions"].sigmoid()
            drivable_area_score = pred_logit["drivable_area_compliance"].sigmoid()
            pdm_score = aggregate_multiplicative_weighted_score(
                multiplicative_metrics=[
                    apply_multiplicative_metric_weight(no_collision_score, self._config.noc),
                    apply_multiplicative_metric_weight(drivable_area_score, self._config.dac),
                ],
                weighted_metrics=[
                    pred_logit["time_to_collision_within_bound"].sigmoid(),
                    pred_logit["ego_progress"].sigmoid(),
                    pred_logit["comfort"].sigmoid(),
                ],
                weighted_metric_weights=[self._config.ttc, self._config.ep, self._config.comfort],
            )
        else:
            pdm_score = (
                self._config.noc * pred_logit["no_at_fault_collisions"].sigmoid().log()
                + self._config.dac * pred_logit["drivable_area_compliance"].sigmoid().log()
                + self._config.ddc * pred_logit["driving_direction_compliance"].sigmoid().log()
                + (
                    self._config.ttc * pred_logit["time_to_collision_within_bound"].sigmoid()
                    + self._config.ep * pred_logit["ego_progress"].sigmoid()
                    + self._config.comfort * pred_logit["comfort"].sigmoid()
                ).log()
            )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]
        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score
        return output
