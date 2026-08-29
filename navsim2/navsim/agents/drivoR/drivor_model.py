# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .score_module.scorer import Scorer
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from .epi_mem_modules import EpiMemoryInjector, EpisodicMemoryBank
from .hd_map_encoder import HDMapEncoder
from navsim.agents.drivoR.utils import pylogger
log = pylogger.get_pylogger(__name__)

class DrivoRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.embed_dims = self._config.tf_d_model

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
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # episodic memory / HD map config
        self.epi_cfg = config["epi_memory"] if "epi_memory" in config else None
        self.use_episodic_memory = bool(self.epi_cfg is not None and self.epi_cfg.get("enabled", False))
        self.hd_map_cfg = config["hd_map"] if "hd_map" in config else None
        self.use_hd_map = bool(self.hd_map_cfg is not None and self.hd_map_cfg.get("enabled", False))
        self.use_separate_epi_hd_map_cross_attn = bool(self.use_episodic_memory and self.use_hd_map)
        extra_cross_attn_count = 2 if self.use_separate_epi_hd_map_cross_attn else 0

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(
            proj_drop=0.1,
            drop_path=0.2,
            config=config,
            num_extra_cross_attns=extra_cross_attn_count,
        )

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(
            num_layers=config.scorer_ref_num,
            d_model=config.tf_d_model,
            proj_drop=0.1,
            drop_path=0.2,
            config=config,
            num_extra_cross_attns=extra_cross_attn_count,
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
        self.epi_memory_injector = None
        self.epi_memory_bank = None
        self.hd_map_mode = "replace"
        self.hd_map_encoder = None
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
                rerank_precomputed_candidates=bank_cfg.get("rerank_precomputed_candidates", False),
                query_pose_noise_longitudinal_std_m=bank_cfg.get("query_pose_noise_longitudinal_std_m", 0.0),
                query_pose_noise_lateral_std_m=bank_cfg.get("query_pose_noise_lateral_std_m", 0.0),
                query_pose_noise_yaw_std_deg=bank_cfg.get("query_pose_noise_yaw_std_deg", 0.0),
                query_pose_noise_seed=bank_cfg.get("query_pose_noise_seed", 0),
                memory_pose_noise_longitudinal_std_m=bank_cfg.get("memory_pose_noise_longitudinal_std_m", 0.0),
                memory_pose_noise_lateral_std_m=bank_cfg.get("memory_pose_noise_lateral_std_m", 0.0),
                memory_pose_noise_yaw_std_deg=bank_cfg.get("memory_pose_noise_yaw_std_deg", 0.0),
                memory_pose_noise_seed=bank_cfg.get("memory_pose_noise_seed", 0),
                yaw_distance_weight=bank_cfg.get("yaw_distance_weight", 0.0),
                exclude_self=bank_cfg.get("exclude_self", True),
                max_distance_m=bank_cfg.get("max_distance_m", None),
                min_yaw=bank_cfg.get("min_yaw", None),
                max_yaw=bank_cfg.get("max_yaw", None),
            )

        if self.use_hd_map:
            self.hd_map_mode = str(self.hd_map_cfg.get("mode", "replace"))
            if self.hd_map_mode not in ("replace", "append"):
                raise ValueError(f"`hd_map.mode` must be one of ['replace', 'append'], got {self.hd_map_mode}.")
            if self.use_separate_epi_hd_map_cross_attn:
                log.info(
                    "Both episodic memory and HD map are enabled; using dedicated cross-attention branches for each "
                    "and preserving the base scene-token branch. `hd_map.mode=%s` is ignored in this mode.",
                    self.hd_map_mode,
                )

            feature_builder_cfg = (
                OmegaConf.to_container(self.hd_map_cfg.feature_builder, resolve=True)
                if "feature_builder" in self.hd_map_cfg and self.hd_map_cfg.feature_builder is not None
                else {}
            )
            encoder_cfg = (
                OmegaConf.to_container(self.hd_map_cfg.encoder, resolve=True)
                if "encoder" in self.hd_map_cfg and self.hd_map_cfg.encoder is not None
                else {}
            )
            encoder_cfg["output_dim"] = config.tf_d_model
            encoder_cfg["points_per_polyline"] = int(feature_builder_cfg.get("points_per_polyline", 20))
            encoder_cfg["query_radius_m"] = float(feature_builder_cfg.get("query_radius_m", 35.0))
            encoder_cfg["max_polylines"] = int(feature_builder_cfg.get("max_polylines", 96))
            self.hd_map_encoder = HDMapEncoder(**encoder_cfg)

        if self.freeze_backbone:
            self._freeze_backbones()

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
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if not self.use_episodic_memory or self.epi_memory_injector is None or self.epi_memory_bank is None:
            return scene_features, None, None, None

        if self.epi_memory_bank.top_k <= 0:
            return scene_features, None, None, None

        if "ego_global_pose" not in features:
            log.warning("Episodic memory enabled but `ego_global_pose` is missing from features.")
            return scene_features, None, None, None

        query_pose = features["ego_global_pose"]
        scenario_tokens = features.get("scenario_token", None)
        epi_tokens, epi_poses = self.epi_memory_bank.query(
            query_global_pose=query_pose,
            scenario_tokens=scenario_tokens,
            device=scene_features.device,
        )
        if self.use_separate_epi_hd_map_cross_attn:
            current_scene_features, scene_padding_mask, epi_features, epi_padding_mask = (
                self.epi_memory_injector.forward_separate(
                    scene_features,
                    epi_tokens,
                    epi_poses,
                )
            )
            return current_scene_features, scene_padding_mask, epi_features, epi_padding_mask

        injected_scene_features = self.epi_memory_injector(scene_features, epi_tokens, epi_poses)

        if isinstance(injected_scene_features, torch.Tensor):
            return injected_scene_features, None, None, None

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
        return torch.stack(padded_features, dim=0), scene_padding_mask, None, None

    @staticmethod
    def _append_scene_features(
        scene_features: torch.Tensor,
        scene_padding_mask: Optional[torch.Tensor],
        extra_features: torch.Tensor,
        extra_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if scene_features.shape[1] == 0:
            return extra_features, extra_padding_mask
        if extra_features.shape[1] == 0:
            return scene_features, scene_padding_mask

        merged_features = torch.cat([scene_features, extra_features], dim=1)
        if scene_padding_mask is None and extra_padding_mask is None:
            return merged_features, None

        batch_size = merged_features.shape[0]
        current_padding_mask = (
            scene_padding_mask
            if scene_padding_mask is not None
            else torch.zeros(
                (batch_size, scene_features.shape[1]), dtype=torch.bool, device=scene_features.device
            )
        )
        extra_padding = (
            extra_padding_mask
            if extra_padding_mask is not None
            else torch.zeros(
                (batch_size, extra_features.shape[1]), dtype=torch.bool, device=extra_features.device
            )
        )
        return merged_features, torch.cat([current_padding_mask, extra_padding], dim=1)

    def _encode_hd_map_features(
        self,
        device: torch.device,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_hd_map or self.hd_map_encoder is None:
            return None, None

        required_keys = [
            "hd_map_coords",
            "hd_map_valid_mask",
            "hd_map_geometry_type_ids",
            "hd_map_element_type_ids",
            "hd_map_on_route_ids",
            "hd_map_has_traffic_light_ids",
            "hd_map_stop_line_subtype_ids",
            "hd_map_speed_limit_mps",
        ]
        if self.hd_map_encoder.include_direction:
            required_keys.append("hd_map_vxvy")
        missing_keys = [key for key in required_keys if key not in features]
        if missing_keys:
            raise ValueError(
                "HD-map encoder is enabled, but required features are missing: "
                + ", ".join(sorted(missing_keys))
                + ". Rebuild the cache with the updated feature builder."
            )

        direction_vectors = (
            features["hd_map_vxvy"].to(device)
            if "hd_map_vxvy" in features
            else torch.zeros_like(features["hd_map_coords"]).to(device)
        )

        map_tokens, map_padding_mask = self.hd_map_encoder(
            coords=features["hd_map_coords"].to(device),
            direction_vectors=direction_vectors,
            valid_mask=features["hd_map_valid_mask"].to(device),
            geometry_type_ids=features["hd_map_geometry_type_ids"].to(device),
            element_type_ids=features["hd_map_element_type_ids"].to(device),
            on_route_ids=features["hd_map_on_route_ids"].to(device),
            has_traffic_light_ids=features["hd_map_has_traffic_light_ids"].to(device),
            stop_line_subtype_ids=features["hd_map_stop_line_subtype_ids"].to(device),
            speed_limit_mps=features["hd_map_speed_limit_mps"].to(device),
        )
        return map_tokens, map_padding_mask

    def _inject_hd_map_features(
        self,
        scene_features: torch.Tensor,
        scene_padding_mask: Optional[torch.Tensor],
        features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        map_tokens, map_padding_mask = self._encode_hd_map_features(scene_features.device, features)
        if map_tokens is None:
            return scene_features, scene_padding_mask

        if self.hd_map_mode == "replace":
            return map_tokens, map_padding_mask

        return self._append_scene_features(
            scene_features=scene_features,
            scene_padding_mask=scene_padding_mask,
            extra_features=map_tokens,
            extra_padding_mask=map_padding_mask,
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ego status and initial traj tokens
        if self._config.full_history_status:
            ego_status: torch.Tensor = features["ego_status"].flatten(-2)
        else:
            ego_status: torch.Tensor = features["ego_status"][:, -1]

        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")

        batch_size = ego_status.shape[0]
        if self.skip_perception_backbone:
            scene_features = ego_token.new_zeros((batch_size, 0, self.embed_dims))
        else:
            scene_features = self.encode_scene_features(features)
        extra_cross_features = None
        extra_cross_padding_masks = None
        scene_features, scene_padding_mask, epi_features, epi_padding_mask = self._inject_episodic_memory(
            scene_features, features
        )
        if self.use_separate_epi_hd_map_cross_attn:
            map_features, map_padding_mask = self._encode_hd_map_features(scene_features.device, features)
            extra_cross_features = [epi_features, map_features]
            extra_cross_padding_masks = [epi_padding_mask, map_padding_mask]
        else:
            scene_features, scene_padding_mask = self._inject_hd_map_features(
                scene_features=scene_features,
                scene_padding_mask=scene_padding_mask,
                features=features,
            )

        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode trajectories
        token_list = self.trajectory_decoder(
            traj_tokens,
            scene_features,
            x_cross_padding_mask=scene_padding_mask,
            extra_cross=extra_cross_features,
            extra_cross_padding_masks=extra_cross_padding_masks,
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
            extra_cross=extra_cross_features,
            extra_cross_padding_masks=extra_cross_padding_masks,
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
