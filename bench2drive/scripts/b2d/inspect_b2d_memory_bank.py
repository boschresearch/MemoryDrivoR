# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "tmp" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
import torch
from PIL import Image
from shapely.geometry import LineString

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.drivoR.epi_mem_modules.memory_bank import EpisodicMemoryBank
from navsim.agents.drivoR.epi_mem_modules.log_group import extract_town_name, normalize_log_group
from scripts.b2d.common import ensure_external_paths
from scripts.b2d.drivor_b2d_config import NameMapping

ensure_external_paths()

B2D_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
)

MAP_LABEL_COLORS = {
    "Broken": "#f4d35e",
    "Solid": "#1b998b",
    "SolidSolid": "#386641",
    "Center": "#f46036",
    "TrafficLight": "#5e60ce",
    "StopSign": "#b5179e",
}

AGENT_COLORS = {
    "car": "#4e79a7",
    "van": "#f28e2b",
    "truck": "#e15759",
    "bicycle": "#76b7b2",
    "traffic_sign": "#edc948",
    "traffic_cone": "#ff9da7",
    "traffic_light": "#b07aa1",
    "pedestrian": "#59a14f",
    "others": "#9c755f",
}

DEFAULT_AGENT_COLOR = "#7f7f7f"
EGO_COLOR = "#111111"
BEV_POINT_CLOUD_RANGE = np.array([-64.0, -64.0, -2.0, 64.0, 64.0, 2.0], dtype=np.float32)
EGO_SIZE_LW = np.array([4.84, 2.30], dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    _, _, zoo_root = ensure_external_paths()
    default_data_root = zoo_root / "data" / "bench2drive"
    default_info_root = zoo_root / "data" / "infos"
    parser = argparse.ArgumentParser(
        description="Visualize B2D query samples and their retrieved episodic memories."
    )
    parser.add_argument("--bank-path", type=Path, required=True)
    parser.add_argument("--bank-ann-files", nargs="+", required=True)
    parser.add_argument("--query-ann-files", nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--map-file", type=Path, default=default_info_root / "b2d_map_infos.pkl")
    parser.add_argument("--output-path", type=Path, default=REPO_ROOT / "tmp" / "b2d_memory_bank_examples.png")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--max-distance-m", type=float, default=None)
    parser.add_argument("--query-tokens", nargs="+", default=None)
    parser.add_argument("--mode", choices=("bev", "camera", "both"), default="bev")
    parser.add_argument("--bev-radius-m", type=float, default=64.0)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _expand_ann_files(inputs: Sequence[str]) -> List[Path]:
    expanded_paths: List[Path] = []
    seen_paths: Set[Path] = set()
    for raw_input in inputs:
        matches = sorted(Path(match).expanduser() for match in glob.glob(os.path.expanduser(raw_input)))
        if len(matches) == 0:
            candidate = Path(raw_input).expanduser()
            if candidate.exists():
                matches = [candidate]
            else:
                raise FileNotFoundError(f"No annotation files matched: {raw_input}")
        for match in matches:
            if match not in seen_paths:
                expanded_paths.append(match)
                seen_paths.add(match)
    return expanded_paths


def _load_info_pkl(path: Path) -> List[Dict]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list of infos in {path}, got {type(data).__name__}")
    return data


def _torch_load_maybe_mmap(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_bank_index_only(bank_path: Path, top_k: int, max_distance_m: Optional[float]) -> EpisodicMemoryBank:
    bank = _torch_load_maybe_mmap(bank_path)
    required = {"global_poses", "tokens"}
    missing = required - set(bank.keys())
    if missing:
        raise KeyError(f"Missing keys in episodic memory bank: {sorted(missing)}")

    memory_bank = EpisodicMemoryBank(
        bank_path=None,
        top_k=top_k,
        use_precomputed_train_neighbors=True,
        max_distance_m=max_distance_m,
    )
    memory_bank.global_poses = torch.as_tensor(bank["global_poses"], dtype=torch.float64)
    memory_bank.tokens = [str(token) for token in bank["tokens"]]
    memory_bank.log_names = [str(log_name) for log_name in bank.get("log_names", [])] or None
    memory_bank.log_groups = (
        [normalize_log_group(log_name) for log_name in memory_bank.log_names]
        if memory_bank.log_names is not None
        else None
    )
    memory_bank.town_names = (
        [extract_town_name(log_name) for log_name in memory_bank.log_names]
        if memory_bank.log_names is not None
        else None
    )

    if "train_neighbors_tokens" in bank:
        memory_bank.train_neighbors_tokens = {
            str(key): [str(value) for value in values]
            for key, values in bank["train_neighbors_tokens"].items()
        }
    elif "train_neighbors" in bank:
        memory_bank.train_neighbors_tokens = {}
        for token, idxs in bank["train_neighbors"].items():
            token_key = str(token)
            mapped_tokens = []
            for idx in idxs:
                resolved_idx = int(idx)
                if 0 <= resolved_idx < len(memory_bank.tokens):
                    mapped_tokens.append(memory_bank.tokens[resolved_idx])
            memory_bank.train_neighbors_tokens[token_key] = mapped_tokens

    memory_bank.token_to_idx = {token: idx for idx, token in enumerate(memory_bank.tokens)}
    return memory_bank


def _make_bank_token(source_name: str, info: Dict) -> str:
    return f"{source_name}:{str(info['folder'])}_{int(info['frame_idx'])}"


def _resolve_query_token_namespace(source_name: str) -> str:
    if source_name.startswith("b2d_infos_"):
        return source_name[len("b2d_infos_") :]
    return source_name


def _make_query_token(source_name: str, raw_index: int) -> str:
    return f"{_resolve_query_token_namespace(source_name)}:{raw_index}"


def _select_evenly_spaced_tokens(tokens: Sequence[str], num_queries: int) -> List[str]:
    token_list = [str(token) for token in tokens]
    if num_queries <= 0 or len(token_list) == 0:
        return []
    if len(token_list) <= num_queries:
        return token_list

    stride = len(token_list) / float(num_queries)
    selected_tokens: List[str] = []
    seen_tokens: Set[str] = set()
    for index in range(num_queries):
        candidate_index = min(int((index + 0.5) * stride), len(token_list) - 1)
        while candidate_index < len(token_list) and token_list[candidate_index] in seen_tokens:
            candidate_index += 1
        if candidate_index >= len(token_list):
            candidate_index = len(token_list) - 1
            while candidate_index >= 0 and token_list[candidate_index] in seen_tokens:
                candidate_index -= 1
        if candidate_index < 0:
            break
        token = token_list[candidate_index]
        selected_tokens.append(token)
        seen_tokens.add(token)
    return selected_tokens


def _query_diversity_key(info: Dict, fallback_group: str) -> tuple[str, str]:
    town_name = _town_key(info) or fallback_group
    sample_name = _query_sample_name(info).split("/")[-1]
    scenario_family = sample_name.split("_Town", 1)[0]
    return town_name, scenario_family


def _scan_query_tokens(
    ann_files: Sequence[Path],
    num_queries: int,
    requested_tokens: Optional[Sequence[str]],
    preferred_tokens: Optional[Iterable[str]],
) -> List[str]:
    if requested_tokens:
        return [str(token) for token in requested_tokens]

    preferred_lookup = None if preferred_tokens is None else set(str(token) for token in preferred_tokens)
    selected_tokens: List[str] = []
    seen_tokens: Set[str] = set()
    fallback_tokens: List[str] = []
    town_to_tokens: Dict[str, List[str]] = {}
    seen_diversity_keys: Set[tuple[str, str]] = set()
    fallback_limit = max(num_queries * 32, num_queries)
    for ann_file in ann_files:
        source_name = ann_file.stem
        for raw_index, _info in enumerate(_load_info_pkl(ann_file)):
            token = _make_query_token(source_name, raw_index)
            if preferred_lookup is not None and token not in preferred_lookup:
                continue
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            if len(fallback_tokens) < fallback_limit:
                fallback_tokens.append(token)

            diversity_key = _query_diversity_key(_info, _resolve_query_token_namespace(source_name))
            if diversity_key in seen_diversity_keys:
                continue
            seen_diversity_keys.add(diversity_key)
            town_bucket = town_to_tokens.setdefault(diversity_key[0], [])
            if len(town_bucket) < num_queries:
                town_bucket.append(token)

    while len(selected_tokens) < num_queries:
        made_progress = False
        for town_bucket in town_to_tokens.values():
            if len(town_bucket) == 0:
                continue
            selected_tokens.append(town_bucket.pop(0))
            made_progress = True
            if len(selected_tokens) >= num_queries:
                break
        if not made_progress:
            break

    if len(selected_tokens) < num_queries:
        for token in _select_evenly_spaced_tokens(fallback_tokens, num_queries):
            if token in selected_tokens:
                continue
            selected_tokens.append(token)
            if len(selected_tokens) >= num_queries:
                break

    return selected_tokens


def _lookup_query_infos(ann_files: Sequence[Path], required_tokens: Sequence[str]) -> Dict[str, Dict]:
    needed_tokens = set(str(token) for token in required_tokens)
    token_to_info: Dict[str, Dict] = {}
    for ann_file in ann_files:
        if not needed_tokens:
            break
        source_name = ann_file.stem
        for raw_index, info in enumerate(_load_info_pkl(ann_file)):
            token = _make_query_token(source_name, raw_index)
            if token in needed_tokens:
                token_to_info[token] = info
                needed_tokens.remove(token)
                if not needed_tokens:
                    break
    if needed_tokens:
        missing_preview = ", ".join(sorted(needed_tokens)[:5])
        raise KeyError(f"Failed to resolve query token(s) from ann files: {missing_preview}")
    return token_to_info


def _lookup_bank_infos(ann_files: Sequence[Path], required_tokens: Sequence[str]) -> Dict[str, Dict]:
    needed_tokens = set(str(token) for token in required_tokens)
    token_to_info: Dict[str, Dict] = {}
    for ann_file in ann_files:
        if not needed_tokens:
            break
        source_name = ann_file.stem
        for info in _load_info_pkl(ann_file):
            token = _make_bank_token(source_name, info)
            if token in needed_tokens:
                token_to_info[token] = info
                needed_tokens.remove(token)
                if not needed_tokens:
                    break
    if needed_tokens:
        missing_preview = ", ".join(sorted(needed_tokens)[:5])
        raise KeyError(f"Failed to resolve bank token(s) from ann files: {missing_preview}")
    return token_to_info


def _info_pose_tensor(info: Dict) -> torch.Tensor:
    return torch.tensor(
        [
            float(info["ego_translation"][0]),
            float(info["ego_translation"][1]),
            float(np.nan_to_num(info["ego_yaw"], nan=np.pi / 2)),
        ],
        dtype=torch.float64,
    )


def _query_sample_name(info: Dict) -> str:
    if "sample_name" in info:
        return str(info["sample_name"])
    if "sample_idx" in info:
        return str(info["sample_idx"])
    if "folder" in info and "frame_idx" in info:
        return f"{info['folder']}_{int(info['frame_idx'])}"
    return "unknown"


def _canonical_name(raw_name: object) -> str:
    return NameMapping.get(str(raw_name), str(raw_name))


def _pose_to_matrix(pose_xyh: torch.Tensor) -> torch.Tensor:
    c = torch.cos(pose_xyh[..., 2])
    s = torch.sin(pose_xyh[..., 2])
    out = torch.zeros(*pose_xyh.shape[:-1], 4, 4, dtype=pose_xyh.dtype)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., 2, 2] = 1.0
    out[..., 3, 3] = 1.0
    out[..., 0, 3] = pose_xyh[..., 0]
    out[..., 1, 3] = pose_xyh[..., 1]
    return out


def _resolve_bank_indices(
    memory_bank: EpisodicMemoryBank,
    query_pose: torch.Tensor,
    query_token: str,
    query_log_name: Optional[str] = None,
) -> List[int]:
    if memory_bank.global_poses is None:
        raise RuntimeError("Memory bank is missing global_poses.")
    query_pose_batch = query_pose[None]
    query_town_name = None
    if query_log_name:
        town_match = __import__("re").search(r"Town(\d+(?:HD)?)", str(query_log_name), flags=__import__("re").IGNORECASE)
        if town_match:
            query_town_name = f"Town{town_match.group(1)}"
    if memory_bank.use_precomputed_train_neighbors and query_token in memory_bank.train_neighbors_tokens:
        idxs = list(memory_bank._precomputed_indices(query_token, query_log_name, query_town_name))
        if len(idxs) < memory_bank.top_k:
            idxs = list(
                memory_bank._online_knn_indices(
                    query_pose_batch,
                    [query_token],
                    [query_log_name],
                    [query_town_name],
                )[0]
            )
    else:
        idxs = list(
            memory_bank._online_knn_indices(
                query_pose_batch,
                [query_token],
                [query_log_name],
                [query_town_name],
            )[0]
        )

    max_distance_sq = (
        float(memory_bank.max_distance_m) * float(memory_bank.max_distance_m)
        if memory_bank.max_distance_m is not None
        else None
    )
    if max_distance_sq is not None and len(idxs) > 0:
        q_xy = query_pose[:2]
        cand_xy = memory_bank.global_poses[idxs, :2]
        d_sq = (cand_xy - q_xy.unsqueeze(0)).square().sum(dim=-1)
        idxs = [idx for idx, keep in zip(idxs, (d_sq <= max_distance_sq).tolist()) if keep]

    use_yaw_filter = memory_bank.min_yaw is not None or memory_bank.max_yaw is not None
    if use_yaw_filter and len(idxs) > 0:
        q_yaw = query_pose[2]
        cand_yaw = memory_bank.global_poses[idxs, 2]
        yaw_delta = q_yaw - cand_yaw
        yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)).abs()
        keep = torch.ones_like(yaw_delta, dtype=torch.bool)
        if memory_bank.min_yaw is not None:
            keep &= yaw_delta >= float(memory_bank.min_yaw)
        if memory_bank.max_yaw is not None:
            keep &= yaw_delta <= float(memory_bank.max_yaw)
        idxs = [idx for idx, is_valid in zip(idxs, keep.tolist()) if is_valid]

    return idxs[: memory_bank.top_k]


def _relative_poses(query_pose: torch.Tensor, memory_poses: torch.Tensor) -> torch.Tensor:
    query_tf = _pose_to_matrix(query_pose[None])
    query_tf_inv = torch.linalg.inv(query_tf)
    mem_tf = _pose_to_matrix(memory_poses)
    return query_tf_inv @ mem_tf


def _relative_distance_yaw(rel_pose: torch.Tensor) -> tuple[float, float]:
    dx = float(rel_pose[0, 3])
    dy = float(rel_pose[1, 3])
    distance_m = math.hypot(dx, dy)
    yaw_deg = math.degrees(math.atan2(float(rel_pose[1, 0]), float(rel_pose[0, 0])))
    return distance_m, yaw_deg


def _select_final_query_tokens(
    candidate_query_tokens: Sequence[str],
    query_info_index: Dict[str, Dict],
    memory_bank: EpisodicMemoryBank,
    requested_query_tokens: Optional[Sequence[str]],
    num_queries: int,
    max_distance_m: Optional[float],
) -> tuple[List[str], Dict[str, List[int]]]:
    query_bank_indices: Dict[str, List[int]] = {}
    strong_tokens: List[str] = []
    weak_tokens: List[str] = []
    min_close_memories = min(memory_bank.top_k, 2) if max_distance_m is not None else 1

    for query_token in candidate_query_tokens:
        query_info = query_info_index[query_token]
        query_pose = _info_pose_tensor(query_info)
        bank_indices = _resolve_bank_indices(memory_bank, query_pose, query_token, str(query_info.get("folder", "")))
        query_bank_indices[query_token] = bank_indices

        if requested_query_tokens is not None:
            continue

        if len(bank_indices) >= min_close_memories:
            strong_tokens.append(query_token)
        elif len(bank_indices) > 0:
            weak_tokens.append(query_token)

    if requested_query_tokens is not None:
        return [str(token) for token in candidate_query_tokens], query_bank_indices

    selected_query_tokens: List[str] = []
    for query_token in strong_tokens:
        selected_query_tokens.append(query_token)
        if len(selected_query_tokens) >= num_queries:
            return selected_query_tokens, query_bank_indices

    for query_token in weak_tokens:
        if query_token in selected_query_tokens:
            continue
        selected_query_tokens.append(query_token)
        if len(selected_query_tokens) >= num_queries:
            break

    if max_distance_m is not None and len(selected_query_tokens) < num_queries:
        print(
            "Selected "
            f"{len(selected_query_tokens)} query examples with nearby memories within {max_distance_m:.1f} m "
            f"(requested {num_queries})."
        )

    return selected_query_tokens, query_bank_indices


def _load_b2d_camera_images(data_root: Path, sensors: Dict) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for camera_name in B2D_CAMERA_ORDER:
        if camera_name not in sensors:
            raise KeyError(f"Missing camera {camera_name} in B2D sensor dictionary.")
        image_path = Path(data_root) / Path(str(sensors[camera_name]["data_path"]))
        with Image.open(image_path) as image_file:
            images.append(np.asarray(image_file.convert("RGB"), dtype=np.uint8))
    return images


def _make_camera_mosaic(images: Sequence[np.ndarray]) -> np.ndarray:
    if len(images) != len(B2D_CAMERA_ORDER):
        raise ValueError(f"Expected {len(B2D_CAMERA_ORDER)} camera images, got {len(images)}")
    top_row = np.concatenate([images[0], images[1]], axis=1)
    bottom_row = np.concatenate([images[2], images[3]], axis=1)
    return np.concatenate([top_row, bottom_row], axis=0)


def _load_map_infos(map_file: Path) -> Dict[str, Dict]:
    if not map_file.exists():
        return {}
    with open(map_file, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in map file {map_file}, got {type(data).__name__}")
    return data


def _ego_size_lw(info: Dict) -> np.ndarray:
    ego_size = info.get("ego_size")
    if ego_size is None:
        return EGO_SIZE_LW.copy()
    ego_size = np.asarray(ego_size, dtype=np.float32)
    if ego_size.shape[0] >= 2:
        return ego_size[:2].copy()
    return EGO_SIZE_LW.copy()


def _box_corners_xy(center_x: float, center_y: float, length: float, width: float, yaw: float) -> np.ndarray:
    half_length = float(length) * 0.5
    half_width = float(width) * 0.5
    base = np.array(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float32,
    )
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    corners = base @ rotation.T
    corners[:, 0] += float(center_x)
    corners[:, 1] += float(center_y)
    return corners


def _draw_heading(ax: Axes, center_x: float, center_y: float, length: float, yaw: float, color: str) -> None:
    heading_length = max(float(length) * 0.7, 1.5)
    dx = heading_length * math.cos(float(yaw))
    dy = heading_length * math.sin(float(yaw))
    ax.plot(
        [center_x, center_x + dx],
        [center_y, center_y + dy],
        color=color,
        linewidth=1.5,
        zorder=5,
    )


def _town_key(info: Dict) -> Optional[str]:
    town_name = info.get("town_name")
    if town_name:
        return str(town_name)
    folder = str(info.get("folder", ""))
    town_match = __import__("re").search(r"Town\d+(?:HD)?", folder, flags=__import__("re").IGNORECASE)
    if town_match:
        return town_match.group(0)
    return None


def _draw_map_layers(ax: Axes, info: Dict, map_infos: Dict[str, Dict], bev_radius_m: float) -> None:
    town_key = _town_key(info)
    if not town_key or town_key not in map_infos:
        return

    map_info = map_infos[town_key]
    world2lidar = np.asarray(info["sensors"]["LIDAR_TOP"]["world2lidar"], dtype=np.float64)

    lane_points = map_info.get("lane_points", [])
    lane_types = map_info.get("lane_types", [])
    for points, lane_type in zip(lane_points, lane_types):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2:
            continue
        homogeneous = np.concatenate([points[:, :3], np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
        local = (world2lidar @ homogeneous.T).T[:, :2]
        mask = (np.abs(local[:, 0]) <= bev_radius_m) & (np.abs(local[:, 1]) <= bev_radius_m)
        local = local[mask]
        if local.shape[0] < 2:
            continue
        color = MAP_LABEL_COLORS.get(str(lane_type), "#c7c7c7")
        ax.plot(local[:, 0], local[:, 1], color=color, linewidth=1.2, alpha=0.85, zorder=1)

    trigger_points = map_info.get("trigger_volumes_points", [])
    trigger_types = map_info.get("trigger_volumes_types", [])
    for points, trigger_type in zip(trigger_points, trigger_types):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        homogeneous = np.concatenate([points[:, :3], np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
        local = (world2lidar @ homogeneous.T).T[:, :2]
        if not np.any((np.abs(local[:, 0]) <= bev_radius_m) & (np.abs(local[:, 1]) <= bev_radius_m)):
            continue
        color = MAP_LABEL_COLORS.get(str(trigger_type), "#b8b8b8")
        polygon = MplPolygon(local[:, :2], closed=True, facecolor=color, edgecolor=color, alpha=0.12, linewidth=1.0, zorder=0)
        ax.add_patch(polygon)


def _draw_agents(ax: Axes, info: Dict) -> None:
    gt_boxes = np.asarray(info.get("gt_boxes", []), dtype=np.float32)
    gt_names = info.get("gt_names", [])
    num_points = np.asarray(info.get("num_points", []))

    if gt_boxes.ndim != 2 or gt_boxes.shape[0] == 0:
        return

    for idx, box in enumerate(gt_boxes):
        if idx < num_points.shape[0] and int(num_points[idx]) == 0:
            continue
        class_name = _canonical_name(gt_names[idx] if idx < len(gt_names) else "others")
        color = AGENT_COLORS.get(class_name, DEFAULT_AGENT_COLOR)
        corners = _box_corners_xy(box[0], box[1], box[3], box[4], box[6])
        patch = MplPolygon(corners, closed=True, facecolor=color, edgecolor=color, alpha=0.28, linewidth=1.4, zorder=3)
        ax.add_patch(patch)
        ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=1.2, zorder=4)
        _draw_heading(ax, float(box[0]), float(box[1]), float(box[3]), float(box[6]), color)


def _draw_ego(ax: Axes, info: Dict) -> None:
    ego_length, ego_width = _ego_size_lw(info)
    corners = _box_corners_xy(0.0, 0.0, float(ego_length), float(ego_width), 0.0)
    patch = MplPolygon(corners, closed=True, facecolor=EGO_COLOR, edgecolor=EGO_COLOR, alpha=0.18, linewidth=1.8, zorder=6)
    ax.add_patch(patch)
    ax.plot(corners[:, 0], corners[:, 1], color=EGO_COLOR, linewidth=1.8, zorder=7)
    _draw_heading(ax, 0.0, 0.0, float(ego_length), 0.0, EGO_COLOR)


def _style_bev_ax(ax: Axes, radius_m: float) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-radius_m, radius_m)
    ax.set_ylim(-radius_m, radius_m)
    ax.set_facecolor("white")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.6)
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _plot_bev(ax: Axes, info: Dict, map_infos: Dict[str, Dict], title: str, bev_radius_m: float) -> None:
    _draw_map_layers(ax, info, map_infos, bev_radius_m)
    _draw_agents(ax, info)
    _draw_ego(ax, info)
    _style_bev_ax(ax, bev_radius_m)
    ax.set_title(title)


def _build_figure(
    query_tokens: Sequence[str],
    query_info_index: Dict[str, Dict],
    query_bank_indices: Dict[str, List[int]],
    bank_info_index: Dict[str, Dict],
    memory_bank: EpisodicMemoryBank,
    data_root: Path,
    map_infos: Dict[str, Dict],
    mode: str,
    bev_radius_m: float,
) -> Figure:
    if memory_bank.global_poses is None:
        raise RuntimeError("Memory bank is missing global_poses.")
    panels_per_entry = 2 if mode == "both" else 1
    max_columns = (1 + memory_bank.top_k) * panels_per_entry
    fig, axes = plt.subplots(
        nrows=len(query_tokens),
        ncols=max_columns,
        figsize=(5.2 * max_columns, 5.0 * len(query_tokens)),
        squeeze=False,
    )

    for row_idx, query_token in enumerate(query_tokens):
        query_info = query_info_index[query_token]
        query_sample_name = _query_sample_name(query_info)
        query_pose = _info_pose_tensor(query_info)
        bank_indices = query_bank_indices[query_token]
        rel_poses = (
            _relative_poses(query_pose, memory_bank.global_poses[bank_indices])
            if len(bank_indices) > 0
            else torch.empty(0, 4, 4, dtype=torch.float64)
        )

        if mode in {"camera", "both"}:
            query_images = _load_b2d_camera_images(data_root, query_info["sensors"])
            camera_ax = axes[row_idx, 0 if mode == "camera" else 0]
            camera_ax.imshow(_make_camera_mosaic(query_images))
            camera_ax.set_title(f"Query Cameras\n{query_token}\n{query_sample_name}")
            camera_ax.axis("off")

        if mode in {"bev", "both"}:
            bev_col = 0 if mode == "bev" else 1
            _plot_bev(
                axes[row_idx, bev_col],
                query_info,
                map_infos,
                f"Query BEV\n{query_token}\n{query_sample_name}",
                bev_radius_m,
            )

        print(f"\nQuery {query_token} sample_name={query_sample_name}")
        for neighbor_rank, bank_idx in enumerate(bank_indices, start=1):
            bank_token = memory_bank.tokens[bank_idx]
            memory_info = bank_info_index[bank_token]
            distance_m, yaw_deg = _relative_distance_yaw(rel_poses[neighbor_rank - 1])
            print(f"  memory {neighbor_rank}: token={bank_token} distance_m={distance_m:.2f} yaw_deg={yaw_deg:+.2f}")

            base_col = neighbor_rank if mode != "both" else neighbor_rank * 2
            if mode in {"camera", "both"}:
                memory_images = _load_b2d_camera_images(data_root, memory_info["sensors"])
                camera_ax = axes[row_idx, base_col if mode == "camera" else base_col]
                camera_ax.imshow(_make_camera_mosaic(memory_images))
                camera_ax.set_title(
                    f"Memory {neighbor_rank} Cameras\n{bank_token}\n{distance_m:.1f} m, {yaw_deg:+.1f} deg"
                )
                camera_ax.axis("off")

            if mode in {"bev", "both"}:
                bev_ax = axes[row_idx, base_col if mode == "bev" else base_col + 1]
                _plot_bev(
                    bev_ax,
                    memory_info,
                    map_infos,
                    f"Memory {neighbor_rank} BEV\n{bank_token}\n{distance_m:.1f} m, {yaw_deg:+.1f} deg",
                    bev_radius_m,
                )

        filled_cols = (1 + len(bank_indices)) if mode != "both" else (1 + len(bank_indices)) * 2
        for empty_col in range(filled_cols, max_columns):
            axes[row_idx, empty_col].axis("off")

    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    bank_ann_files = _expand_ann_files(args.bank_ann_files)
    query_ann_files = _expand_ann_files(args.query_ann_files)

    memory_bank = _load_bank_index_only(args.bank_path, int(args.top_k), args.max_distance_m)
    map_infos = _load_map_infos(args.map_file)

    preferred_query_tokens = memory_bank.train_neighbors_tokens.keys() if memory_bank.train_neighbors_tokens else None
    candidate_query_count = int(args.num_queries)
    if args.query_tokens is None and args.max_distance_m is not None:
        candidate_query_count = max(int(args.num_queries) * 32, int(args.num_queries))

    candidate_query_tokens = _scan_query_tokens(
        ann_files=query_ann_files,
        num_queries=candidate_query_count,
        requested_tokens=args.query_tokens,
        preferred_tokens=preferred_query_tokens,
    )
    if len(candidate_query_tokens) == 0:
        raise RuntimeError("No query tokens available for visualization.")

    query_info_index = _lookup_query_infos(query_ann_files, candidate_query_tokens)
    query_tokens, query_bank_indices = _select_final_query_tokens(
        candidate_query_tokens=candidate_query_tokens,
        query_info_index=query_info_index,
        memory_bank=memory_bank,
        requested_query_tokens=args.query_tokens,
        num_queries=int(args.num_queries),
        max_distance_m=args.max_distance_m,
    )
    if len(query_tokens) == 0:
        raise RuntimeError("No query tokens with matching memories available for visualization.")

    query_info_index = {query_token: query_info_index[query_token] for query_token in query_tokens}

    required_bank_tokens: List[str] = []
    for query_token in query_tokens:
        bank_indices = query_bank_indices[query_token]
        required_bank_tokens.extend(memory_bank.tokens[bank_idx] for bank_idx in bank_indices)

    bank_info_index = _lookup_bank_infos(bank_ann_files, required_bank_tokens)

    fig = _build_figure(
        query_tokens=query_tokens,
        query_info_index=query_info_index,
        query_bank_indices=query_bank_indices,
        bank_info_index=bank_info_index,
        memory_bank=memory_bank,
        data_root=args.data_root,
        map_infos=map_infos,
        mode=args.mode,
        bev_radius_m=float(args.bev_radius_m),
    )

    fig.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=160)
    print(f"\nSaved visualization to {args.output_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
