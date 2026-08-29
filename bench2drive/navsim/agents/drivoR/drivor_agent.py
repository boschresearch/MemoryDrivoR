# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from contextlib import contextmanager
from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from .drivor_model import DrivoRModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, LearningRateMonitor
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from .drivor_features import DrivoRTargetBuilder
from .drivor_features import DrivoRFeatureBuilder
import sys
from omegaconf import OmegaConf
import math

class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - train {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - val {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(self, pl_module)
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics = dict()
        val_metrics = dict()
        other_metrics = dict()
        for k,v in metrics.items():
            if "train/" in k:
                train_metrics[k]=v
            elif "val/" in k:
                val_metrics[k]=v
            else:
                other_metrics[k]=v
        print(f"\n###########  Epoch {trainer.current_epoch} ##########")
        for k,v in train_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in val_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in other_metrics.items():
            print(f"{k},{v:.3f}")
        print(f"###########\n")

class DrivoRAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr_args: dict,
            checkpoint_path: str = None,
            b2d_cache_path: str = None,
            loss: nn.Module = None,
            progress_bar: bool = True,
            scheduler_args: dict = None,
            batch_size: int = 64,
            num_gpus: int = 1,
            cache_data: bool = False,
    ):
        super().__init__()
        self._config = config
        self._lr_args = lr_args
        self._checkpoint_path = checkpoint_path
        self._b2d_cache_path = b2d_cache_path
        self.progress_bar = progress_bar
        self.scheduler_args = scheduler_args
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self._cache_data = cache_data
        self.b2d = bool(config.get("b2d", False))
        self.enable_collision_gate_for_ep = bool(config.get("enable_collision_gate_for_ep", True))
        self.enable_drivable_gate_for_ep = bool(config.get("enable_drivable_gate_for_ep", True))

        if not self._cache_data:
            self._drivor_model = DrivoRModel(config)

        if not self._cache_data and self._checkpoint_path == "": # only for training
            self.bce_logit_loss = nn.BCEWithLogitsLoss()

            self.ray=True

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                self.worker = RayDistributedNoTorch(threads_per_node=8)
                self.worker_map=worker_map


            if self.b2d:
                cache_root = Path(self._b2d_cache_path) if self._b2d_cache_path else Path(os.getenv("NAVSIM_EXP_ROOT")) / "B2d_cache"
                self.train_metric_cache_paths = load_feature_target_from_pickle(cache_root / "train_fut_boxes.gz")
                self.test_metric_cache_paths = load_feature_target_from_pickle(cache_root / "val_fut_boxes.gz")
                self.train_metric_cache_paths_synthetic = None
                self.test_metric_cache_paths_synthetic = None

                from .score_module.compute_b2d_score import get_scores
                self.get_scores = get_scores

                with open(Path(os.getenv("NAVSIM_EXP_ROOT")) / "map.pkl", "rb") as f:
                    self.map_infos = pickle.load(f)
                self.cuda_map = False
            else:
                from .score_module.compute_navsim_score import get_scores

                metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
                try:
                    # add synthetic metric_cache
                    metric_cache_synthetic_0 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-0"))
                    metric_cache_synthetic_1 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-1"))
                    metric_cache_synthetic_2 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-2"))
                    metric_cache_synthetic_3 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-3"))
                    metric_cache_synthetic_4 = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_synthetic_reaction_pdm_v1.0-4"))

                    self.train_metric_cache_paths_synthetic = metric_cache_synthetic_0.metric_cache_paths
                    self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_0.metric_cache_paths)
                    self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_1.metric_cache_paths)
                    self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_2.metric_cache_paths)
                    self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_3.metric_cache_paths)
                    self.train_metric_cache_paths_synthetic.update(metric_cache_synthetic_4.metric_cache_paths)

                    self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
                except:
                    self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic = None

                self.test_metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
                self.train_metric_cache_paths = metric_cache.metric_cache_paths
                self.test_metric_cache_paths = metric_cache.metric_cache_paths

                self.get_scores = get_scores

            self.loss = loss
            


    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"]
            remapped_state_dict = {
                k.replace("agent._drivor_model", "_drivor_model"): v for k, v in state_dict.items()
            }
            incompatible_keys = self.load_state_dict(remapped_state_dict, strict=False)
            if incompatible_keys.missing_keys:
                raise RuntimeError(
                    "Checkpoint is missing required DrivoRAgent parameters: "
                    + ", ".join(incompatible_keys.missing_keys)
                )

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        # return SensorConfig(
        #     cam_f0=[3],
        #     cam_l0=[3],
        #     cam_l1=[],
        #     cam_l2=[],
        #     cam_r0=[3],
        #     cam_r1=[],
        #     cam_r2=[],
        #     cam_b0=[3],
        #     lidar_pc=[],
        # )
        return SensorConfig(
            cam_f0=OmegaConf.to_object(self._config["cam_f0"]),
            cam_l0=OmegaConf.to_object(self._config["cam_l0"]),
            cam_l1=OmegaConf.to_object(self._config["cam_l1"]),
            cam_l2=OmegaConf.to_object(self._config["cam_l2"]),
            cam_r0=OmegaConf.to_object(self._config["cam_r0"]),
            cam_r1=OmegaConf.to_object(self._config["cam_r1"]),
            cam_r2=OmegaConf.to_object(self._config["cam_r2"]),
            cam_b0=OmegaConf.to_object(self._config["cam_b0"]),
            lidar_pc=OmegaConf.to_object(self._config["lidar_pc"]),
        )
    
    def get_target_builders(self) :
        return [DrivoRTargetBuilder(config=self._config)]

    def get_feature_builders(self) :
        return [DrivoRFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._cache_data:
            raise RuntimeError("DrivoRAgent was initialized in cache_data mode and cannot run forward().")
        return self._drivor_model(features)

    @contextmanager
    def override_skip_perception_backbone(self, enabled: bool):
        if self._cache_data:
            yield
            return

        previous_cfg_value = bool(self._config.get("skip_perception_backbone", False))
        previous_model_value = self._drivor_model.skip_perception_backbone
        self._config["skip_perception_backbone"] = enabled
        self._drivor_model.skip_perception_backbone = enabled
        try:
            yield
        finally:
            self._config["skip_perception_backbone"] = previous_cfg_value
            self._drivor_model.skip_perception_backbone = previous_model_value

    def compute_score(self, targets, proposals, test=True):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
            metric_cache_paths_synthetic = self.train_metric_cache_paths_synthetic
        else:
            metric_cache_paths = self.test_metric_cache_paths
            metric_cache_paths_synthetic = self.test_metric_cache_paths_synthetic

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()

        if self.b2d:
            data_points = []
            lidar2worlds = targets["lidar2world"]
            all_proposals = torch.cat([proposals, target_trajectory[:, None]], dim=1)
            all_proposals_xy = all_proposals[:, :, :, :2]
            all_proposals_heading = all_proposals[:, :, :, 2:]

            all_pos = all_proposals_xy.reshape(len(target_trajectory), -1, 2)
            mid_points = (all_pos.amax(1) + all_pos.amin(1)) / 2
            dists = torch.linalg.norm(all_pos - mid_points[:, None], dim=-1).amax(1) + 5

            xyz = torch.cat(
                [mid_points[..., :2], torch.zeros_like(mid_points[..., :1]), torch.ones_like(mid_points[..., :1])],
                dim=-1,
            )
            xys = torch.einsum("nij,nj->ni", lidar2worlds, xyz)[:, :2]

            vel = torch.cat(
                [all_proposals_xy[:, :, :1], all_proposals_xy[:, :, 1:] - all_proposals_xy[:, :, :-1]], dim=2
            ) / 0.5

            proposals_05 = torch.cat([all_proposals_xy + vel * 0.5, all_proposals_heading], dim=-1)
            proposals_1 = torch.cat([all_proposals_xy + vel * 1.0, all_proposals_heading], dim=-1)
            proposals_ttc = torch.stack([all_proposals, proposals_05, proposals_1], dim=3)

            from .score_module.compute_b2d_score import compute_corners_torch

            ego_corners_ttc = compute_corners_torch(proposals_ttc.reshape(-1, 3)).reshape(
                proposals_ttc.shape[0], proposals_ttc.shape[1], proposals_ttc.shape[2], proposals_ttc.shape[3], 4, 2
            )

            ego_corners_center = torch.cat([ego_corners_ttc[:, :, :, 0], all_proposals_xy[:, :, :, None]], dim=-2)
            ego_corners_center_xyz = torch.cat(
                [
                    ego_corners_center,
                    torch.zeros_like(ego_corners_center[..., :1]),
                    torch.ones_like(ego_corners_center[..., :1]),
                ],
                dim=-1,
            )
            global_ego_corners_centers = torch.einsum(
                "nij,nptkj->nptki", lidar2worlds, ego_corners_center_xyz
            )[..., :2]

            accs = torch.linalg.norm(vel[:, :, 1:] - vel[:, :, :-1], dim=-1) / 0.5
            turning_rate = torch.abs(
                torch.cat(
                    [all_proposals_heading[:, :, :1, 0] - np.pi / 2, all_proposals_heading[:, :, 1:, 0] - all_proposals_heading[:, :, :-1, 0]],
                    dim=2,
                )
            ) / 0.5
            comforts = (accs[:, :-1] < accs[:, -1:].max()).all(-1) & (
                turning_rate[:, :-1] < turning_rate[:, -1:].max()
            ).all(-1)

            if not self.cuda_map:
                for key, value in self.map_infos.items():
                    self.map_infos[key] = torch.tensor(value).to(target_trajectory.device)
                self.cuda_map = True

            for token, town_name, proposal, target_traj, comfort, dist, xy, global_corners, local_corners in zip(
                targets["token"],
                targets["town_name"],
                proposals.cpu().numpy(),
                target_trajectory.cpu().numpy(),
                comforts.cpu().numpy(),
                dists.cpu().numpy(),
                xys,
                global_ego_corners_centers,
                ego_corners_ttc.cpu().numpy(),
            ):
                all_lane_points = self.map_infos[town_name[:6]]
                dist_to_cur = torch.linalg.norm(all_lane_points[:, :2] - xy, dim=-1)
                nearby_point = all_lane_points[dist_to_cur < dist]

                lane_xy = nearby_point[:, :2]
                lane_width = nearby_point[:, 2]
                lane_id = nearby_point[:, -1]

                dist_to_lane = torch.linalg.norm(global_corners[None] - lane_xy[:, None, None, None], dim=-1)
                on_road = dist_to_lane < lane_width[:, None, None, None]
                on_road_all = on_road.any(0).all(-1)

                nearest_lane = torch.argmin(dist_to_lane - lane_width[:, None, None, None], dim=0)
                nearest_lane_id = lane_id[nearest_lane]
                center_nearest_lane_id = nearest_lane_id[:, :, -1]
                nearest_road_id = torch.round(center_nearest_lane_id)
                target_road_id = torch.unique(nearest_road_id[-1])
                on_route_all = torch.isin(nearest_road_id, target_road_id)

                corner_nearest_lane_id = nearest_lane_id[:, :, :-1]
                batch_multiple_lanes_mask = (corner_nearest_lane_id != corner_nearest_lane_id[..., :1]).any(-1)
                on_road_all = on_road_all == on_road_all[-1:]
                ego_areas = torch.stack([batch_multiple_lanes_mask, on_road_all, on_route_all], dim=-1)

                data_points.append(
                    {
                        "fut_box_corners": metric_cache_paths[token],
                        "_ego_coords": local_corners,
                        "target_traj": target_traj,
                        "proposal": proposal,
                        "comfort": comfort,
                        "ego_areas": ego_areas.cpu().numpy(),
                        "noc_weight": float(self._config.noc),
                        "dac_weight": float(self._config.dac),
                        "ttc_weight": float(self._config.ttc),
                        "ep_weight": float(self._config.ep),
                        "comfort_weight": float(self._config.comfort),
                        "enable_collision_gate_for_ep": self.enable_collision_gate_for_ep,
                        "enable_drivable_gate_for_ep": self.enable_drivable_gate_for_ep,
                    }
                )
        else:
            data_points = [
                {
                    "token": metric_cache_paths[token] if token in metric_cache_paths else metric_cache_paths_synthetic[token],
                    "poses": poses,
                    "test": test
                }
                for token, poses in zip(targets["token"], proposals.cpu().numpy())
            ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.loss(targets, pred, self._config, self.compute_score)

    def get_optimizers(self):

        global_batchsize = self.batch_size * self.num_gpus
        lr_scale = math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
        base_lr = float(self._lr_args["base_lr"])
        lr = base_lr * lr_scale

        epi_memory_lr = None
        epi_memory_base_lr = self._lr_args.get("epi_memory_base_lr", None)
        if epi_memory_base_lr is not None:
            epi_memory_base_lr = float(epi_memory_base_lr)
            if epi_memory_base_lr < 0.0:
                raise ValueError(f"lr_args.epi_memory_base_lr must be >= 0.0, got {epi_memory_base_lr}")
            epi_memory_lr = epi_memory_base_lr * lr_scale

        if self._lr_args["name"] == "Adam":
            optimizer_cls = torch.optim.Adam
        elif self._lr_args["name"] == "AdamW":
            optimizer_cls = torch.optim.AdamW
        else:
            raise NotImplementedError

        trainable_params = [param for param in self._drivor_model.parameters() if param.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found in DrivoR model.")

        if epi_memory_lr is None:
            optimizer = optimizer_cls(trainable_params, lr=lr)
        else:
            epi_injector = getattr(self._drivor_model, "epi_memory_injector", None)
            assert epi_injector is not None, "`lr_args.epi_memory_base_lr` is set, but model has no `epi_memory_injector`."

            # Keep non-pretrainable parameters on the base LR.
            base_lr_param_names = {"curr_flag_emb", "epi_flag_emb"}
            epi_param_ids = {
                id(param)
                for name, param in epi_injector.named_parameters()
                if param.requires_grad
                and not name.startswith("output_projector.")
                and name not in base_lr_param_names
            }
            epi_params = [param for param in trainable_params if id(param) in epi_param_ids]
            non_epi_params = [param for param in trainable_params if id(param) not in epi_param_ids]

            assert len(epi_params), (
                "`lr_args.epi_memory_base_lr` is set, but no trainable pretrainable episodic-memory "
                "injector parameters were found."
            )

            param_groups = []
            if len(non_epi_params) > 0:
                param_groups.append({"params": non_epi_params, "lr": lr})
            param_groups.append({"params": epi_params, "lr": epi_memory_lr})
            optimizer = optimizer_cls(param_groups, lr=lr)

        if self.scheduler_args is not None:

            T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) *  self.scheduler_args.num_epochs)
            T_max_ramp = int(T_max * 0.1)
            T_max_cosine = max(T_max - T_max_ramp, 1)

            def lr_lambda(step: int) -> float:
                if T_max <= 0:
                    return 1.0
                if T_max_ramp > 0 and step < T_max_ramp:
                    ramp_progress = step / max(T_max_ramp, 1)
                    return 1e-6 + (1.0 - 1e-6) * ramp_progress

                cosine_step = max(step - T_max_ramp, 0)
                cosine_progress = min(cosine_step / T_max_cosine, 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        
        else:
            return optimizer

    def get_training_callbacks(self):

        checkpoint_cb_best = ModelCheckpoint(save_top_k=1,
                                        monitor='val/score_epoch',
                                        filename='best-{epoch}-{step}',
                                        mode="max"
                                        )
        
        checkpoint_cb = ModelCheckpoint(save_last=True)

        lr_monitor = LearningRateMonitor(logging_interval="step", 
                                            log_momentum=False,
                                            log_weight_decay=False)
        
        if self.progress_bar:
            return [checkpoint_cb_best, checkpoint_cb, lr_monitor]
        else:
            progress_bar = LitProgressBar()
            return [checkpoint_cb_best, checkpoint_cb, progress_bar, lr_monitor]
