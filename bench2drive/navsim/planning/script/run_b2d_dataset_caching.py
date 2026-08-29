# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.planning.training.dataset import dump_feature_target_to_pickle
from scripts.b2d.common import ensure_external_paths

ensure_external_paths()

from mmcv.datasets.B2D_vad_dataset import B2D_VAD_Dataset
from scripts.b2d.drivor_b2d_config import (
    B2D_CAMERA_ORDER,
    NameMapping,
    build_b2d_ego_status_vector,
    class_names,
    eval_cfg,
    load_b2d_camera_images,
    modality,
    point_cloud_range,
    preprocess_b2d_camera_images,
)

def _parse_split_ann_file_args(split_ann_file_args):
    split_ann_files = []
    for entry in split_ann_file_args:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --split-ann-files entry `{entry}`. Expected format `split=/path/to/info.pkl`."
            )
        split_name, ann_file = entry.split("=", 1)
        split_name = split_name.strip()
        ann_file_path = Path(ann_file).expanduser()
        if not split_name:
            raise ValueError(f"Invalid --split-ann-files entry `{entry}`: empty split name.")
        if not ann_file_path.exists():
            raise FileNotFoundError(f"B2D info file not found: {ann_file_path}")
        split_ann_files.append((split_name, ann_file_path))
    return split_ann_files


def compute_corners(boxes):
    x = boxes[:, 0]
    y = boxes[:, 1]
    half_width = boxes[:, 2] / 2
    half_length = boxes[:, 3] / 2
    headings = boxes[:, 4]

    cos_yaw = np.cos(headings)[..., None]
    sin_yaw = np.sin(headings)[..., None]

    corners_x = np.stack([half_length, -half_length, -half_length, half_length], axis=-1)
    corners_y = np.stack([half_width, half_width, -half_width, -half_width], axis=-1)

    rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
    rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y
    return np.stack((rot_corners_x + x[..., None], rot_corners_y + y[..., None]), axis=-1)


class CacheB2DDataset(B2D_VAD_Dataset):
    def __init__(self, split, ann_file, data_root, map_file, cache_root, image_size):
        super().__init__(
            point_cloud_range=point_cloud_range,
            queue_length=1,
            data_root=str(data_root),
            ann_file=str(ann_file),
            eval_cfg=eval_cfg,
            map_file=str(map_file),
            pipeline=[],
            name_mapping=NameMapping,
            modality=modality,
            classes=class_names,
        )
        self.split = split
        self.cache_root = Path(cache_root) / split
        self.data_root_path = Path(data_root)
        self.image_size = tuple(int(dimension) for dimension in image_size)

    def _get_filtered_annotations(self, index):
        ann_info = self.get_ann_info(index)
        gt_bboxes_3d = ann_info["gt_bboxes_3d"]
        gt_labels_3d = ann_info["gt_labels_3d"]
        gt_attr_labels = ann_info["attr_labels"]

        bev_range = np.asarray(point_cloud_range)[[0, 1, 3, 4]]
        range_mask = gt_bboxes_3d.in_range_bev(bev_range)
        range_mask_np = range_mask.cpu().numpy().astype(bool)

        gt_bboxes_3d = gt_bboxes_3d[range_mask]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        gt_labels_3d = gt_labels_3d[range_mask_np]
        gt_attr_labels = gt_attr_labels[range_mask_np]

        class_mask = gt_labels_3d >= 0
        gt_bboxes_3d = gt_bboxes_3d[class_mask]
        gt_attr_labels = gt_attr_labels[class_mask]
        return gt_bboxes_3d, gt_attr_labels

    def get_fut_box(self, gt_agent_feats, gt_agent_boxes, horizon=6):
        agent_feats = torch.as_tensor(gt_agent_feats, dtype=torch.float32)
        agent_boxes = torch.as_tensor(gt_agent_boxes, dtype=torch.float32).clone()
        agent_num = int(agent_feats.shape[0])

        gt_agent_fut_trajs = agent_feats[..., : horizon * 2].reshape(-1, horizon, 2)
        gt_agent_fut_mask = agent_feats[..., horizon * 2 : horizon * 3].reshape(-1, horizon)
        gt_agent_fut_yaw = agent_feats[..., horizon * 3 + 10 : horizon * 4 + 10].reshape(-1, horizon, 1)

        gt_agent_fut_trajs = torch.cumsum(gt_agent_fut_trajs, dim=1)
        gt_agent_fut_yaw = torch.cumsum(gt_agent_fut_yaw, dim=1)

        agent_boxes[:, 6:7] = -1 * (agent_boxes[:, 6:7] + np.pi / 2)
        gt_agent_fut_trajs = gt_agent_fut_trajs + agent_boxes[:, None, 0:2]
        gt_agent_fut_yaw = gt_agent_fut_yaw + agent_boxes[:, None, 6:7]

        x = gt_agent_fut_trajs[:, :, 0]
        y = gt_agent_fut_trajs[:, :, 1]
        yaw = gt_agent_fut_yaw[:, :, 0]
        agent_width = agent_boxes[:, None, 3].repeat(1, horizon)
        agent_length = agent_boxes[:, None, 4].repeat(1, horizon)

        fut_boxes = torch.stack([x, y, agent_width, agent_length, yaw], dim=-1)
        fut_boxes = fut_boxes * gt_agent_fut_mask[:, :, None]
        corners = compute_corners(fut_boxes.cpu().numpy().reshape(-1, 5)).reshape(agent_num, horizon, 4, 2)
        return corners.astype(np.float32)

    def _get_ego_target_trajectory(self, index, ego_fut_trajs):
        target_xy = torch.from_numpy(np.asarray(ego_fut_trajs)).cumsum(dim=0)
        if target_xy.ndim != 2 or target_xy.shape[1] != 2:
            raise ValueError(
                f"Expected ego_fut_trajs with shape (num_poses, 2), got {tuple(target_xy.shape)} at index {index}."
            )

        current_info = self.data_infos[index]
        current_world2lidar = np.asarray(
            current_info["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float64
        )
        headings = []
        for future_index in range(
            index + self.sample_interval,
            index + (self.future_frames + 1) * self.sample_interval,
            self.sample_interval,
        ):
            if future_index >= len(self.data_infos) or self.data_infos[future_index]["folder"] != current_info["folder"]:
                raise ValueError(f"Missing valid future ego pose for dataset index {index}.")

            future_world2lidar = np.asarray(
                self.data_infos[future_index]["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float64
            )
            future_to_current_lidar = current_world2lidar @ np.linalg.inv(future_world2lidar)
            relative_yaw = np.arctan2(future_to_current_lidar[1, 0], future_to_current_lidar[0, 0])
            headings.append(relative_yaw + np.pi / 2)

        if len(headings) != target_xy.shape[0]:
            raise ValueError(
                f"Expected {target_xy.shape[0]} future headings, got {len(headings)} at index {index}."
            )

        target_heading = torch.as_tensor(headings, dtype=target_xy.dtype).unsqueeze(-1)
        return torch.cat([target_xy, target_heading], dim=-1)

    def __getitem__(self, idx):
        token = f"{self.split}:{idx}"
        data = self.get_data_info(idx)
        if data is None:
            return {token: None}

        if not data["fut_valid_flag"]:
            return {token: None}

        gt_bboxes_3d, gt_attr_labels = self._get_filtered_annotations(idx)
        gt_agent_boxes = gt_bboxes_3d.tensor
        fut_boxes = self.get_fut_box(gt_attr_labels, gt_agent_boxes)

        ann_info = self.data_infos[idx]
        ego_vel = float(ann_info["ego_vel"][0])
        ego_accel = ann_info["ego_accel"][:2]
        ego_translation = ann_info["ego_translation"]

        command_near_xy = np.array(
            [
                ann_info["command_near_xy"][0] - ego_translation[0],
                ann_info["command_near_xy"][1] - ego_translation[1],
            ]
        )
        yaw = ann_info["ego_yaw"]
        theta_to_lidar = -(yaw - np.pi / 2)
        rotation_matrix = np.array(
            [[np.cos(theta_to_lidar), -np.sin(theta_to_lidar)], [np.sin(theta_to_lidar), np.cos(theta_to_lidar)]]
        )
        local_command_xy = rotation_matrix @ command_near_xy

        ego_status = build_b2d_ego_status_vector(
            ego_vel,
            ego_accel,
            local_command_xy,
            data["ego_fut_cmd"],
            warning_context=f"B2D cache ego_status token={token}",
        )[None]

        camera_images = load_b2d_camera_images(self.data_root_path, ann_info["sensors"], B2D_CAMERA_ORDER)
        camera_feature = preprocess_b2d_camera_images(camera_images, self.image_size)

        token_path = self.cache_root / token
        os.makedirs(token_path, exist_ok=True)
        dump_feature_target_to_pickle(
            token_path / "drivor_feature.gz",
            {
                "ego_status": ego_status,
                "camera_feature": camera_feature,
                "ego_global_pose": torch.tensor(
                    [[float(ego_translation[0]), float(ego_translation[1]), float(yaw)]],
                    dtype=torch.float64,
                ),
                "scenario_token": token,
                "scenario_log_name": str(ann_info["folder"]),
            },
        )

        target_traj = self._get_ego_target_trajectory(idx, data["ego_fut_trajs"])

        world2lidar = np.array(ann_info["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float32)
        dump_feature_target_to_pickle(
            token_path / "drivor_target.gz",
            {
                "trajectory": target_traj.to(torch.float32),
                "token": token,
                "town_name": ann_info["town_name"],
                "lidar2world": np.linalg.inv(world2lidar).astype(np.float32),
            },
        )

        return {token: fut_boxes}


def _collate_identity(batch):
    return batch


def main():
    parser = argparse.ArgumentParser(description="Cache Bench2Drive data into DrivoR cache artifacts.")
    default_cache_path = None
    navsim_exp_root = os.environ.get("NAVSIM_EXP_ROOT")
    if navsim_exp_root:
        default_cache_path = Path(navsim_exp_root) / "B2d_cache"
    parser.add_argument("--cache-path", type=Path, default=default_cache_path)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--map-file", type=Path, default=None)
    parser.add_argument(
        "--split-ann-files",
        nargs="+",
        default=None,
        help="Entries in the form `split=/path/to/info.pkl`. Defaults to train/val under the resolved info root.",
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    args = parser.parse_args()
    if args.cache_path is None:
        raise EnvironmentError("Set NAVSIM_EXP_ROOT or pass --cache-path explicitly.")

    _, _, bench2drivezoo_root = ensure_external_paths()
    agent_cfg = OmegaConf.load(
        REPO_ROOT / "navsim/planning/script/config/common/agent/drivoR_b2d.yaml"
    )
    image_size = agent_cfg.config.image_size

    data_root = args.data_root if args.data_root is not None else bench2drivezoo_root / "data" / "bench2drive"
    info_root = bench2drivezoo_root / "data" / "infos"
    map_file = args.map_file if args.map_file is not None else info_root / "b2d_map_infos.pkl"
    split_ann_files = (
        _parse_split_ann_file_args(args.split_ann_files)
        if args.split_ann_files is not None
        else [
            ("train", info_root / "b2d_infos_train.pkl"),
            ("val", info_root / "b2d_infos_val.pkl"),
        ]
    )
    os.makedirs(args.cache_path, exist_ok=True)

    for split, ann_file in split_ann_files:
        dataset = CacheB2DDataset(
            split=split,
            ann_file=ann_file,
            data_root=data_root,
            map_file=map_file,
            cache_root=args.cache_path,
            image_size=image_size,
        )

        fut_box = {}
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            pin_memory=False,
            collate_fn=_collate_identity,
        )

        for batch in tqdm(dataloader, desc=f"Caching {split}"):
            for key, value in batch[0].items():
                if value is not None:
                    fut_box[key] = value

        dump_feature_target_to_pickle(args.cache_path / f"{split}_fut_boxes.gz", fut_box)


if __name__ == "__main__":
    main()
