# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.b2d.common import ensure_external_paths

ensure_external_paths()

from mmcv.datasets.B2D_vad_dataset import B2D_VAD_Dataset

from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.agents.drivoR.epi_mem_modules.log_group import extract_town_name, normalize_log_group
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

logger = logging.getLogger(__name__)

B2D_WEATHER_BUCKETS = {
    "easy": {0, 1, 7, 26},
    "okay": {2, 3, 5, 6, 15, 18},
    "medium": {8, 14},
    "hard": {9, 10, 11, 12, 13, 19, 22, 23},
    "extreme": {20, 21, 25},
}

_B2D_WEATHER_TAG_RE = re.compile(r"(?i)weather[_-]?(\d+)")
_B2D_SAVE_NAME_WEATHER_RE = re.compile(r"_(\d+)_(\d{2}_\d{2}_\d{2}_\d{2}_\d{2})$")


def _strip_torchrun_local_rank_arg() -> None:
    cleaned_argv = [sys.argv[0]]
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--local-rank", "--local_rank"}:
            skip_next = True
            continue
        if arg.startswith("--local-rank=") or arg.startswith("--local_rank="):
            continue
        cleaned_argv.append(arg)
    sys.argv = cleaned_argv


_strip_torchrun_local_rank_arg()


@dataclass(frozen=True)
class B2DBankRecord:
    raw_index: int
    token: str
    log_name: str
    timestamp_us: int
    pose_xyh: Tuple[float, float, float]
    ego_fut_cmd: Tuple[float, ...]
    info: Dict


class B2DBankFeatureDataset(Dataset):
    def __init__(
        self,
        records: Sequence[B2DBankRecord],
        data_root: Path,
        image_size: Sequence[int],
    ) -> None:
        super().__init__()
        self._records = list(records)
        self._data_root = Path(data_root)
        self._image_size = tuple(int(v) for v in image_size)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        record = self._records[idx]
        info = record.info

        ego_translation = np.asarray(info["ego_translation"], dtype=np.float64)
        ego_accel = np.asarray(info["ego_accel"][:2], dtype=np.float32)
        command_near_xy = np.asarray(
            [
                float(info["command_near_xy"][0]) - float(ego_translation[0]),
                float(info["command_near_xy"][1]) - float(ego_translation[1]),
            ],
            dtype=np.float64,
        )
        yaw = float(np.nan_to_num(info["ego_yaw"], nan=np.pi / 2))
        theta_to_lidar = -(yaw - np.pi / 2)
        rotation_matrix = np.asarray(
            [
                [np.cos(theta_to_lidar), -np.sin(theta_to_lidar)],
                [np.sin(theta_to_lidar), np.cos(theta_to_lidar)],
            ],
            dtype=np.float64,
        )
        local_command_xy = rotation_matrix @ command_near_xy

        ego_status = build_b2d_ego_status_vector(
            float(info["ego_vel"][0]),
            ego_accel,
            local_command_xy,
            record.ego_fut_cmd,
            warning_context=f"B2D memory-bank ego_status token={record.token}",
        )[None]

        try:
            camera_images = load_b2d_camera_images(self._data_root, info["sensors"], B2D_CAMERA_ORDER)
        except FileNotFoundError as exc:
            logger.warning("Skipping B2D bank sample token=%s because an image file is missing: %s", record.token, exc)
            return None
        camera_feature = preprocess_b2d_camera_images(camera_images, self._image_size)

        features = {
            "ego_status": ego_status,
            "camera_feature": camera_feature,
            "ego_global_pose": torch.tensor([record.pose_xyh], dtype=torch.float64),
            "scenario_token": record.token,
            "scenario_log_name": record.log_name,
            "__dataset_index__": torch.tensor(idx, dtype=torch.long),
        }
        return features, {}


def _init_distributed_context() -> Tuple[int, int, int]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world_size, local_rank


def _get_batch_size(features: Dict) -> int:
    for value in features.values():
        if torch.is_tensor(value):
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    return 0


def _to_device_features(features: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in features.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _collate_skip_missing_bank_samples(batch):
    kept_batch = [item for item in batch if item is not None]
    if len(kept_batch) == 0:
        return None
    return default_collate(kept_batch)


def _load_drivor_checkpoint_if_provided(model: torch.nn.Module, checkpoint_path: Optional[str]) -> None:
    if not checkpoint_path:
        logger.warning("No checkpoint provided; episodic memory bank will use current model weights.")
        return

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading DrivoR checkpoint from %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    def strip_prefix(key: str) -> str:
        for prefix in ("agent._drivor_model.", "_drivor_model.", "drivor_model."):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    mapped = {strip_prefix(k): v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(mapped, strict=False)
    logger.info(
        "Checkpoint load: missing=%d unexpected=%d",
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )


def _compute_precomputed_neighbors(
    tokens: List[str],
    log_names: List[str],
    global_poses: torch.Tensor,
    k: int,
    yaw_distance_weight: float = 0.0,
    chunk_size: int = 1024,
) -> Dict[str, List[str]]:
    if k <= 0:
        return {}

    n_samples = global_poses.shape[0]
    if len(tokens) != n_samples or len(log_names) != n_samples:
        raise ValueError("Token/log lengths must match global_poses length.")
    if n_samples <= 1:
        return {token: [] for token in tokens}

    log_groups = [normalize_log_group(log_name) for log_name in log_names]
    unique_log_groups = set(log_groups)
    max_cross_log_neighbors = max(len(unique_log_groups) - 1, 0)
    if max_cross_log_neighbors == 0:
        logger.warning(
            "Only one unique source-log group found while building precomputed neighbors; all neighbor lists will be empty."
        )
        return {token: [] for token in tokens}

    k = min(k, n_samples - 1)
    target_k = min(k, max_cross_log_neighbors)
    if k > max_cross_log_neighbors:
        logger.warning(
            "Requested top_k=%d but only %d cross-log neighbors are possible with %d unique logs.",
            k,
            max_cross_log_neighbors,
            len(unique_log_groups),
        )

    global_poses = global_poses.cpu()
    all_xy = global_poses[:, :2]
    all_yaw = global_poses[:, 2]
    neighbors: Dict[str, List[str]] = {}

    from scipy.spatial import cKDTree  # type: ignore

    tree = cKDTree(all_xy.numpy())
    cand_k = max(k + 1, 64, k * 32)
    if yaw_distance_weight > 0:
        cand_k = max(cand_k, 128, k * 64)
    cand_k = min(cand_k, n_samples)

    log_group_to_id = {log_group: idx for idx, log_group in enumerate(sorted(unique_log_groups))}
    log_ids = [log_group_to_id[log_group] for log_group in log_groups]
    underfilled_count = 0

    for start in tqdm(range(0, n_samples, chunk_size), desc="Precomputing KNN"):
        end = min(start + chunk_size, n_samples)
        _, idx = tree.query(all_xy[start:end].numpy(), k=cand_k)
        if cand_k == 1:
            idx = idx[:, None]

        idx_t = torch.as_tensor(idx, dtype=torch.long)
        if yaw_distance_weight > 0:
            q_xy = all_xy[start:end, None, :]
            cand_xy = all_xy[idx_t, :]
            d_xy = (q_xy - cand_xy).square().sum(dim=-1)

            q_yaw = all_yaw[start:end, None]
            cand_yaw = all_yaw[idx_t]
            yaw_delta = q_yaw - cand_yaw
            yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
            d_all = d_xy + yaw_distance_weight * yaw_delta.square()
            rank_order = torch.argsort(d_all, dim=-1)
            ranked_idx = idx_t.gather(1, rank_order)
        else:
            ranked_idx = idx_t

        for row_idx in range(end - start):
            self_global_idx = start + row_idx
            token = tokens[self_global_idx]
            query_log_id = log_ids[self_global_idx]
            chosen: List[int] = []
            used_neighbor_log_ids = set()

            for cand_idx in ranked_idx[row_idx].tolist():
                cand_idx = int(cand_idx)
                if cand_idx == self_global_idx:
                    continue
                cand_log_id = log_ids[cand_idx]
                if cand_log_id == query_log_id or cand_log_id in used_neighbor_log_ids:
                    continue
                chosen.append(cand_idx)
                used_neighbor_log_ids.add(cand_log_id)
                if len(chosen) >= k:
                    break

            neighbors[token] = [tokens[v] for v in chosen]
            if len(chosen) < target_k:
                underfilled_count += 1

    if underfilled_count > 0:
        logger.warning(
            "%d/%d samples had fewer than %d precomputed cross-log-group neighbors.",
            underfilled_count,
            n_samples,
            target_k,
        )

    return neighbors


def _merge_shard_payloads(shard_payloads: List[Dict]) -> Dict:
    all_indices: List[int] = []
    all_tokens: List[str] = []
    all_log_names: List[str] = []
    all_timestamps_us: List[int] = []
    memory_parts: List[torch.Tensor] = []
    pose_parts: List[torch.Tensor] = []

    for shard_payload in tqdm(
        shard_payloads,
        desc="Validating shard payloads",
        disable=(len(shard_payloads) <= 1),
    ):
        shard_tokens = [str(token) for token in shard_payload.get("tokens", [])]
        shard_indices = [int(idx) for idx in shard_payload.get("sample_indices", [])]
        shard_log_names = [str(log_name) for log_name in shard_payload.get("log_names", [])]
        shard_timestamps_us = [int(ts) for ts in shard_payload.get("timestamps_us", [])]

        if len(shard_tokens) == 0:
            continue
        if not (
            len(shard_tokens)
            == len(shard_indices)
            == len(shard_log_names)
            == len(shard_timestamps_us)
        ):
            raise ValueError("Distributed shard payload has inconsistent list lengths.")

        shard_memory = torch.as_tensor(shard_payload["memory_tokens"], dtype=torch.float32).contiguous()
        shard_poses = torch.as_tensor(shard_payload["global_poses"], dtype=torch.float64).contiguous()
        if shard_memory.ndim != 3:
            raise ValueError(f"Shard memory_tokens must be [N, T, D], got {tuple(shard_memory.shape)}")
        if shard_poses.ndim != 2 or shard_poses.shape[1] != 3:
            raise ValueError(f"Shard global_poses must be [N, 3], got {tuple(shard_poses.shape)}")
        if shard_memory.shape[0] != len(shard_tokens) or shard_poses.shape[0] != len(shard_tokens):
            raise ValueError("Distributed shard payload has inconsistent tensor/list lengths.")

        all_indices.extend(shard_indices)
        all_tokens.extend(shard_tokens)
        all_log_names.extend(shard_log_names)
        all_timestamps_us.extend(shard_timestamps_us)
        memory_parts.append(shard_memory)
        pose_parts.append(shard_poses)

    if len(memory_parts) == 0:
        raise RuntimeError("No samples extracted while building the B2D episodic memory bank.")
    if len(set(all_indices)) != len(all_indices):
        raise RuntimeError("Distributed shard merge found duplicate sample indices.")

    memory_tokens = torch.cat(memory_parts, dim=0).contiguous()
    global_poses = torch.cat(pose_parts, dim=0).contiguous()
    sort_order = torch.argsort(torch.as_tensor(all_indices, dtype=torch.long))
    sort_order_list = sort_order.tolist()

    return {
        "tokens": [all_tokens[idx] for idx in sort_order_list],
        "log_names": [all_log_names[idx] for idx in sort_order_list],
        "timestamps_us": [all_timestamps_us[idx] for idx in sort_order_list],
        "global_poses": global_poses[sort_order],
        "memory_tokens": memory_tokens[sort_order],
    }


def _wait_for_shard_files(
    shard_dir: Path,
    world_size: int,
    timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
) -> None:
    expected_paths = [shard_dir / f"rank_{shard_rank:04d}.pt" for shard_rank in range(world_size)]
    start_time = time.perf_counter()
    last_log_bucket = -1

    while True:
        missing_paths = [path for path in expected_paths if not path.exists()]
        if len(missing_paths) == 0:
            logger.info("All %d shard files are available for merge.", world_size)
            return

        elapsed = time.perf_counter() - start_time
        if elapsed >= timeout_seconds:
            missing_preview = ", ".join(str(path.name) for path in missing_paths[:10])
            raise TimeoutError(
                f"Timed out after {timeout_seconds:.1f}s waiting for shard files in {shard_dir}. "
                f"Missing {len(missing_paths)}/{world_size}: {missing_preview}"
            )

        current_log_bucket = int(elapsed // 60)
        if current_log_bucket != last_log_bucket:
            logger.info(
                "Waiting for shard files: %d/%d available after %.1fs.",
                world_size - len(missing_paths),
                world_size,
                elapsed,
            )
            last_log_bucket = current_log_bucket

        time.sleep(poll_interval_seconds)


def _resolve_timestamp_us(info: Dict) -> int:
    if "timestamp" in info and info["timestamp"] is not None:
        timestamp = float(info["timestamp"])
        if timestamp > 1e12:
            return int(round(timestamp))
        return int(round(timestamp * 1e6))

    if "frame_idx" not in info:
        raise KeyError("B2D info record is missing both `timestamp` and `frame_idx`.")
    return int(int(info["frame_idx"]) * 100000)


def _resolve_allowed_weather_indices(
    weather_levels: Optional[Sequence[str]],
    weather_indices: Optional[Sequence[int]],
) -> Optional[set[int]]:
    allowed_weather_indices: set[int] = set()

    if weather_levels is not None:
        for weather_level in weather_levels:
            normalized_weather_level = str(weather_level).strip().lower()
            if normalized_weather_level not in B2D_WEATHER_BUCKETS:
                valid_levels = ", ".join(sorted(B2D_WEATHER_BUCKETS.keys()))
                raise ValueError(
                    f"Unknown weather level `{weather_level}`. Valid levels: {valid_levels}."
                )
            allowed_weather_indices.update(B2D_WEATHER_BUCKETS[normalized_weather_level])

    if weather_indices is not None:
        allowed_weather_indices.update(int(weather_index) for weather_index in weather_indices)

    if len(allowed_weather_indices) == 0:
        return None
    return allowed_weather_indices


def _extract_weather_index_from_folder(folder_name: str) -> int:
    folder_basename = Path(folder_name).name

    weather_match = _B2D_WEATHER_TAG_RE.search(folder_basename)
    if weather_match is not None:
        return int(weather_match.group(1))

    save_name_match = _B2D_SAVE_NAME_WEATHER_RE.search(folder_basename)
    if save_name_match is not None:
        return int(save_name_match.group(1))

    raise ValueError(
        f"Could not parse weather index from B2D folder `{folder_name}`. "
        "Expected either a `weatherXX` tag or the standard Bench2Drive save-name suffix."
    )


def _resolve_weather_index(info: Dict) -> int:
    for key in ("weather_id", "weather_idx", "weather"):
        if key in info and info[key] is not None:
            return int(info[key])
    return _extract_weather_index_from_folder(str(info["folder"]))


def _build_b2d_dataset_from_ann_file(
    ann_file: Path,
    data_root: Path,
    map_file: Path,
) -> B2D_VAD_Dataset:
    if not ann_file.exists():
        raise FileNotFoundError(f"B2D info file not found: {ann_file}")

    return B2D_VAD_Dataset(
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


def _resolve_split_ann_files(info_root: Path, splits: Sequence[str]) -> List[Path]:
    ann_files = [info_root / f"b2d_infos_{split}.pkl" for split in splits]
    missing_ann_files = [ann_file for ann_file in ann_files if not ann_file.exists()]
    if missing_ann_files:
        missing_str = ", ".join(str(path) for path in missing_ann_files)
        raise FileNotFoundError(f"B2D info file(s) not found: {missing_str}")
    return ann_files


def _normalize_ann_files(ann_files: Sequence[Path]) -> List[Path]:
    normalized_ann_files: List[Path] = []
    for ann_file in ann_files:
        normalized_ann_file = Path(ann_file)
        if not normalized_ann_file.exists():
            raise FileNotFoundError(f"B2D info file not found: {normalized_ann_file}")
        normalized_ann_files.append(normalized_ann_file)
    return normalized_ann_files


def _make_bank_token(source_name: str, info: Dict) -> str:
    folder = str(info["folder"])
    frame_idx = int(info["frame_idx"])
    return f"{source_name}:{folder}_{frame_idx}"


def _resolve_query_token_namespace(source_name: str) -> str:
    if source_name.startswith("b2d_infos_"):
        return source_name[len("b2d_infos_") :]
    return source_name


def _make_query_token(source_name: str, raw_index: int) -> str:
    return f"{_resolve_query_token_namespace(source_name)}:{raw_index}"


def _build_selected_records_for_dataset(
    dataset: B2D_VAD_Dataset,
    source_name: str,
    token_mode: str,
    allowed_weather_indices: Optional[set[int]],
    min_translation_m: Optional[float],
    max_interval_s: Optional[float],
    min_translation_noise_std: float,
    show_progress: bool,
) -> List[B2DBankRecord]:
    sorted_indices = sorted(
        range(len(dataset.data_infos)),
        key=lambda idx: (str(dataset.data_infos[idx]["folder"]), int(dataset.data_infos[idx]["frame_idx"])),
    )

    use_translation = min_translation_m is not None and float(min_translation_m) > 0.0
    use_time = max_interval_s is not None and float(max_interval_s) > 0.0
    translation_threshold_m = max(float(min_translation_m or 0.0), 0.0)
    max_interval_s = None if not use_time else float(max_interval_s)
    noise_std_m = max(float(min_translation_noise_std), 0.0)
    keep_all_valid = not use_translation and max_interval_s is None

    selected_records: List[B2DBankRecord] = []
    last_kept_xy_by_log: Dict[str, np.ndarray] = {}
    last_kept_timestamp_us_by_log: Dict[str, int] = {}
    valid_count = 0
    weather_filtered_count = 0

    for raw_index in tqdm(
        sorted_indices,
        desc=f"Selecting B2D samples ({source_name})",
        disable=(not show_progress),
    ):
        info = dataset.data_infos[raw_index]
        if allowed_weather_indices is not None:
            weather_index = _resolve_weather_index(info)
            if weather_index not in allowed_weather_indices:
                weather_filtered_count += 1
                continue
        _, _ego_fut_trajs, ego_fut_masks, command = dataset.get_ego_trajs(
            raw_index,
            dataset.sample_interval,
            dataset.past_frames,
            dataset.future_frames,
        )
        if not bool((ego_fut_masks == 1).all()):
            continue
        valid_count += 1

        route_name = str(info["folder"])
        timestamp_us = _resolve_timestamp_us(info)
        pose_xyh = (
            float(info["ego_translation"][0]),
            float(info["ego_translation"][1]),
            float(np.nan_to_num(info["ego_yaw"], nan=np.pi / 2)),
        )

        keep_due_translation = False
        if route_name in last_kept_xy_by_log and use_translation:
            noisy_threshold_m = translation_threshold_m
            if noise_std_m > 0.0:
                noisy_threshold_m = max(noisy_threshold_m + float(torch.randn(()).item()) * noise_std_m, 0.0)
            distance_m = float(np.linalg.norm(np.asarray(pose_xyh[:2]) - last_kept_xy_by_log[route_name]))
            keep_due_translation = distance_m >= noisy_threshold_m

        keep_due_time = False
        if route_name in last_kept_timestamp_us_by_log and max_interval_s is not None:
            elapsed_s = float(timestamp_us - last_kept_timestamp_us_by_log[route_name]) / 1e6
            keep_due_time = elapsed_s >= max_interval_s

        if keep_all_valid or route_name not in last_kept_xy_by_log or keep_due_translation or keep_due_time:
            if token_mode == "bank":
                token = _make_bank_token(source_name=source_name, info=info)
            elif token_mode == "query":
                token = _make_query_token(source_name=source_name, raw_index=raw_index)
            else:
                raise ValueError(f"Unsupported token_mode: {token_mode}")
            selected_records.append(
                B2DBankRecord(
                    raw_index=raw_index,
                    token=token,
                    log_name=route_name,
                    timestamp_us=timestamp_us,
                    pose_xyh=pose_xyh,
                    ego_fut_cmd=tuple(float(value) for value in np.asarray(command).reshape(-1).tolist()),
                    info=info,
                )
            )
            last_kept_xy_by_log[route_name] = np.asarray(pose_xyh[:2], dtype=np.float64)
            last_kept_timestamp_us_by_log[route_name] = timestamp_us

    logger.info(
        "Selected %d/%d valid B2D samples for source=%s after weather_filter_skips=%d with min_translation_m=%s, max_interval_s=%s, min_translation_noise_std=%.3f.",
        len(selected_records),
        valid_count,
        source_name,
        weather_filtered_count,
        f"{translation_threshold_m:.3f}" if use_translation else "disabled",
        f"{max_interval_s:.3f}" if max_interval_s is not None else "disabled",
        noise_std_m,
    )
    return selected_records


def _build_records_from_ann_files(
    ann_files: Sequence[Path],
    data_root: Path,
    map_file: Path,
    token_mode: str,
    allowed_weather_indices: Optional[set[int]],
    min_translation_m: Optional[float],
    max_interval_s: Optional[float],
    min_translation_noise_std: float,
    show_progress: bool,
) -> List[B2DBankRecord]:
    all_records: List[B2DBankRecord] = []

    for ann_file in ann_files:
        source_name = ann_file.stem
        split_dataset = _build_b2d_dataset_from_ann_file(
            ann_file=ann_file,
            data_root=data_root,
            map_file=map_file,
        )
        split_records = _build_selected_records_for_dataset(
            dataset=split_dataset,
            source_name=source_name,
            token_mode=token_mode,
            allowed_weather_indices=allowed_weather_indices,
            min_translation_m=min_translation_m,
            max_interval_s=max_interval_s,
            min_translation_noise_std=min_translation_noise_std,
            show_progress=show_progress,
        )
        all_records.extend(split_records)

    logger.info("Selected %d total B2D records across ann_files=%s.", len(all_records), [str(path) for path in ann_files])
    return all_records


def _assert_unique_tokens(records: Sequence[B2DBankRecord], name: str) -> None:
    seen_tokens = set()
    duplicate_tokens = set()
    for record in records:
        if record.token in seen_tokens:
            duplicate_tokens.add(record.token)
        else:
            seen_tokens.add(record.token)
    if duplicate_tokens:
        duplicate_preview = ", ".join(sorted(list(duplicate_tokens))[:5])
        raise RuntimeError(f"Found duplicate {name} token(s): {duplicate_preview}")


def _parse_args() -> argparse.Namespace:
    _, _, zoo_root = ensure_external_paths()
    default_info_root = zoo_root / "data" / "infos"
    default_data_root = zoo_root / "data" / "bench2drive"
    default_map_file = default_info_root / "b2d_map_infos.pkl"
    default_agent_config = REPO_ROOT / "navsim" / "planning" / "script" / "config" / "common" / "agent" / "drivoR_b2d.yaml"
    default_output_path = None
    navsim_exp_root = os.environ.get("NAVSIM_EXP_ROOT")
    if navsim_exp_root:
        default_output_path = Path(navsim_exp_root) / "episodic_bank" / "b2d" / "train" / "bank.pt"

    parser = argparse.ArgumentParser(description="Build a DrivoR episodic memory bank directly from Bench2Drive info PKLs.")
    parser.add_argument("--agent-config", type=Path, default=default_agent_config)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=default_output_path, required=(default_output_path is None))
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--info-root", type=Path, default=default_info_root)
    parser.add_argument("--map-file", type=Path, default=default_map_file)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--bank-ann-files", nargs="+", type=Path, default=None)
    parser.add_argument("--query-ann-files", nargs="+", type=Path, default=None)
    parser.add_argument("--weather-levels", nargs="+", default=None)
    parser.add_argument("--weather-indices", nargs="+", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--yaw-distance-weight", type=float, default=0.0)
    parser.add_argument("--precompute-chunk-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--min-translation-m", type=float, default=None)
    parser.add_argument("--max-interval-s", type=float, default=None)
    parser.add_argument("--min-translation-noise-std", type=float, default=0.0)
    parser.add_argument("--shard-wait-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--disable-precomputed-neighbors", action="store_true")
    return parser.parse_args()


def _compute_asymmetric_precomputed_neighbors(
    query_tokens: List[str],
    query_log_names: List[str],
    query_global_poses: torch.Tensor,
    bank_tokens: List[str],
    bank_log_names: List[str],
    bank_global_poses: torch.Tensor,
    k: int,
    yaw_distance_weight: float = 0.0,
    chunk_size: int = 1024,
) -> Dict[str, List[str]]:
    if k <= 0:
        return {}

    n_queries = query_global_poses.shape[0]
    n_bank = bank_global_poses.shape[0]
    if len(query_tokens) != n_queries or len(query_log_names) != n_queries:
        raise ValueError("Query token/log lengths must match query_global_poses length.")
    if len(bank_tokens) != n_bank or len(bank_log_names) != n_bank:
        raise ValueError("Bank token/log lengths must match bank_global_poses length.")
    if n_bank == 0:
        return {token: [] for token in query_tokens}

    bank_log_groups = [normalize_log_group(log_name) for log_name in bank_log_names]
    bank_town_names = [extract_town_name(log_name) for log_name in bank_log_names]
    unique_bank_log_groups = {log_group for log_group in bank_log_groups if log_group}
    if not unique_bank_log_groups:
        logger.warning(
            "No valid source-log groups found while building asymmetric precomputed neighbors; all neighbor lists will be empty."
        )
        return {token: [] for token in query_tokens}
    town_to_bank_log_groups: Dict[str, set[str]] = {}
    for bank_log_group, bank_town_name in zip(bank_log_groups, bank_town_names):
        if not bank_town_name or not bank_log_group:
            continue
        town_to_bank_log_groups.setdefault(bank_town_name, set()).add(bank_log_group)

    k = min(k, n_bank)
    query_global_poses = query_global_poses.cpu()
    bank_global_poses = bank_global_poses.cpu()
    bank_xy = bank_global_poses[:, :2]
    bank_yaw = bank_global_poses[:, 2]
    neighbors: Dict[str, List[str]] = {}

    from scipy.spatial import cKDTree  # type: ignore

    global_bank_indices = torch.arange(n_bank, dtype=torch.long)
    global_tree = cKDTree(bank_xy.numpy())
    town_to_bank_indices: Dict[str, torch.Tensor] = {}
    town_to_bank_xy: Dict[str, torch.Tensor] = {}
    town_to_bank_yaw: Dict[str, torch.Tensor] = {}
    town_to_tree: Dict[str, cKDTree] = {}
    for bank_town_name in sorted(set(bank_town_names)):
        if not bank_town_name:
            continue
        town_indices = [idx for idx, cand_town_name in enumerate(bank_town_names) if cand_town_name == bank_town_name]
        if len(town_indices) == 0:
            continue
        town_index_tensor = torch.as_tensor(town_indices, dtype=torch.long)
        town_xy = bank_xy[town_index_tensor]
        town_to_bank_indices[bank_town_name] = town_index_tensor
        town_to_bank_xy[bank_town_name] = town_xy
        town_to_bank_yaw[bank_town_name] = bank_yaw[town_index_tensor]
        town_to_tree[bank_town_name] = cKDTree(town_xy.numpy())
    cand_k = max(k, 64, k * 32)
    if yaw_distance_weight > 0:
        cand_k = max(cand_k, 128, k * 64)
    cand_k = min(cand_k, n_bank)
    underfilled_count = 0

    def _get_bank_subset(
        query_town_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, cKDTree, int]:
        if query_town_name and query_town_name in town_to_bank_indices:
            town_indices = town_to_bank_indices[query_town_name]
            return (
                town_indices,
                town_to_bank_xy[query_town_name],
                town_to_bank_yaw[query_town_name],
                town_to_tree[query_town_name],
                int(town_indices.numel()),
            )
        return global_bank_indices, bank_xy, bank_yaw, global_tree, n_bank

    def _select_cross_log_diverse_indices(
        ranked_candidates: Sequence[int],
        query_log_group: str,
        query_town_name: str,
    ) -> List[int]:
        chosen: List[int] = []
        used_neighbor_log_groups = set()
        for cand_idx in ranked_candidates:
            cand_idx = int(cand_idx)
            if query_town_name and bank_town_names[cand_idx] != query_town_name:
                continue
            cand_log_group = bank_log_groups[cand_idx]
            if cand_log_group:
                if query_log_group and cand_log_group == query_log_group:
                    continue
                if cand_log_group in used_neighbor_log_groups:
                    continue
                used_neighbor_log_groups.add(cand_log_group)
            chosen.append(cand_idx)
            if len(chosen) >= k:
                break
        return chosen

    for start in tqdm(range(0, n_queries, chunk_size), desc="Precomputing asymmetric KNN"):
        end = min(start + chunk_size, n_queries)
        chunk_query_poses = query_global_poses[start:end]
        chunk_query_town_names = [extract_town_name(query_log_names[start + row_idx]) for row_idx in range(end - start)]
        chunk_ranked_idx: List[torch.Tensor] = [torch.empty(0, dtype=torch.long) for _ in range(end - start)]
        town_to_chunk_rows: Dict[str, List[int]] = {}
        for row_idx, town_name in enumerate(chunk_query_town_names):
            town_to_chunk_rows.setdefault(town_name, []).append(row_idx)

        for query_town_name, chunk_rows in town_to_chunk_rows.items():
            bank_subset_indices, bank_subset_xy, bank_subset_yaw, subset_tree, subset_size = _get_bank_subset(query_town_name)
            if subset_size == 0:
                continue

            local_cand_k = min(cand_k, subset_size)
            chunk_row_tensor = torch.as_tensor(chunk_rows, dtype=torch.long)
            _, local_idx = subset_tree.query(chunk_query_poses[chunk_row_tensor, :2].numpy(), k=local_cand_k)
            if local_cand_k == 1:
                local_idx = local_idx[:, None]

            local_idx_t = torch.as_tensor(local_idx, dtype=torch.long)
            global_idx_t = bank_subset_indices[local_idx_t]
            if yaw_distance_weight > 0:
                q_xy = chunk_query_poses[chunk_row_tensor, None, :2]
                cand_xy = bank_subset_xy[local_idx_t, :]
                d_xy = (q_xy - cand_xy).square().sum(dim=-1)

                q_yaw = chunk_query_poses[chunk_row_tensor, None, 2]
                cand_yaw = bank_subset_yaw[local_idx_t]
                yaw_delta = q_yaw - cand_yaw
                yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
                d_all = d_xy + yaw_distance_weight * yaw_delta.square()
                rank_order = torch.argsort(d_all, dim=-1)
                ranked_idx = global_idx_t.gather(1, rank_order)
            else:
                ranked_idx = global_idx_t

            for local_row_idx, chunk_row_idx in enumerate(chunk_rows):
                chunk_ranked_idx[chunk_row_idx] = ranked_idx[local_row_idx]

        for row_idx in range(end - start):
            query_global_idx = start + row_idx
            query_log_group = normalize_log_group(query_log_names[query_global_idx])
            query_town_name = chunk_query_town_names[row_idx]
            bank_subset_indices, bank_subset_xy, bank_subset_yaw, subset_tree, subset_size = _get_bank_subset(query_town_name)
            available_log_groups = (
                town_to_bank_log_groups.get(query_town_name, unique_bank_log_groups)
                if not query_town_name
                else town_to_bank_log_groups.get(query_town_name, set())
            )
            available_cross_log_groups = len(
                available_log_groups - ({query_log_group} if query_log_group else set())
            )
            target_k = min(k, max(available_cross_log_groups, 0))
            chosen = _select_cross_log_diverse_indices(chunk_ranked_idx[row_idx].tolist(), query_log_group, query_town_name)
            initial_subset_cand_k = min(cand_k, subset_size)
            retry_cand_k = min(max(initial_subset_cand_k * 4, k * 128, 256), subset_size)
            if len(chosen) < target_k and retry_cand_k > initial_subset_cand_k:
                _, retry_idx = subset_tree.query(query_global_poses[query_global_idx, :2].numpy(), k=retry_cand_k)
                retry_idx_t = torch.as_tensor(retry_idx, dtype=torch.long).reshape(1, -1)
                retry_global_idx_t = bank_subset_indices[retry_idx_t]
                if yaw_distance_weight > 0:
                    q_xy = query_global_poses[query_global_idx : query_global_idx + 1, None, :2]
                    cand_xy = bank_subset_xy[retry_idx_t, :]
                    d_xy = (q_xy - cand_xy).square().sum(dim=-1)

                    q_yaw = query_global_poses[query_global_idx : query_global_idx + 1, None, 2]
                    cand_yaw = bank_subset_yaw[retry_idx_t]
                    yaw_delta = q_yaw - cand_yaw
                    yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
                    d_all = d_xy + yaw_distance_weight * yaw_delta.square()
                    rank_order = torch.argsort(d_all, dim=-1)
                    retry_ranked_idx = retry_global_idx_t.gather(1, rank_order)[0]
                else:
                    retry_ranked_idx = retry_global_idx_t[0]
                chosen = _select_cross_log_diverse_indices(
                    retry_ranked_idx.tolist(),
                    query_log_group,
                    query_town_name,
                )

            neighbors[query_tokens[query_global_idx]] = [bank_tokens[v] for v in chosen]
            if len(chosen) < target_k:
                underfilled_count += 1

    if underfilled_count > 0:
        logger.warning(
            "%d/%d queries had fewer than the available cross-log-group asymmetric neighbors.",
            underfilled_count,
            n_queries,
        )

    return neighbors


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    rank, world_size, local_rank = _init_distributed_context()

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    if args.max_interval_s is not None and args.max_interval_s <= 0:
        raise ValueError("--max-interval-s must be positive when provided.")
    if args.min_translation_m is not None and args.min_translation_m < 0:
        raise ValueError("--min-translation-m must be non-negative when provided.")
    if args.min_translation_noise_std < 0:
        raise ValueError("--min-translation-noise-std must be non-negative.")

    logger.info(
        "Starting B2D episodic memory extraction on rank=%d, local_rank=%d, world_size=%d",
        rank,
        local_rank,
        world_size,
    )

    agent_omega = OmegaConf.load(args.agent_config)
    agent_cfg = agent_omega.config
    image_size = agent_cfg.image_size

    model_cfg = OmegaConf.create(OmegaConf.to_container(agent_cfg, resolve=True))
    if "epi_memory" in model_cfg and model_cfg.epi_memory is not None:
        model_cfg.epi_memory.enabled = False

    model = DrivoRModel(model_cfg)
    _load_drivor_checkpoint_if_provided(model, str(args.checkpoint_path) if args.checkpoint_path else None)

    bank_ann_files = (
        _normalize_ann_files(args.bank_ann_files)
        if args.bank_ann_files
        else _resolve_split_ann_files(info_root=args.info_root, splits=args.splits)
    )
    query_ann_files = _normalize_ann_files(args.query_ann_files) if args.query_ann_files else None
    allowed_bank_weather_indices = _resolve_allowed_weather_indices(args.weather_levels, args.weather_indices)
    if allowed_bank_weather_indices is not None:
        logger.info(
            "Filtering bank-side B2D samples to weather indices=%s",
            sorted(allowed_bank_weather_indices),
        )

    selected_records = _build_records_from_ann_files(
        ann_files=bank_ann_files,
        data_root=args.data_root,
        map_file=args.map_file,
        token_mode="bank",
        allowed_weather_indices=allowed_bank_weather_indices,
        min_translation_m=args.min_translation_m,
        max_interval_s=args.max_interval_s,
        min_translation_noise_std=args.min_translation_noise_std,
        show_progress=(rank == 0),
    )
    _assert_unique_tokens(selected_records, "bank")
    full_dataset = B2DBankFeatureDataset(
        records=selected_records,
        data_root=args.data_root,
        image_size=image_size,
    )
    if len(full_dataset) == 0:
        raise RuntimeError("B2D episodic memory selection produced an empty dataset.")

    query_records: Optional[List[B2DBankRecord]] = None
    if query_ann_files is not None:
        query_records = _build_records_from_ann_files(
            ann_files=query_ann_files,
            data_root=args.data_root,
            map_file=args.map_file,
            token_mode="query",
            allowed_weather_indices=None,
            min_translation_m=None,
            max_interval_s=None,
            min_translation_noise_std=0.0,
            show_progress=(rank == 0),
        )
        _assert_unique_tokens(query_records, "query")

    if world_size > 1:
        rank_indices = list(range(rank, len(full_dataset), world_size))
        dataset = Subset(full_dataset, rank_indices)
    else:
        rank_indices = list(range(len(full_dataset)))
        dataset = full_dataset

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=_collate_skip_missing_bank_samples,
        shuffle=False,
        drop_last=False,
    )

    if torch.cuda.is_available():
        if world_size > 1:
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    all_tokens: List[str] = []
    all_log_names: List[str] = []
    all_timestamps_us: List[int] = []
    all_sample_indices: List[int] = []
    memory_chunks: List[torch.Tensor] = []
    pose_chunks: List[torch.Tensor] = []
    token_to_record = {record.token: record for record in selected_records}

    with torch.no_grad():
        for batch in tqdm(
            loader,
            desc=f"Extracting B2D scene features (rank {rank})",
            disable=(rank != 0),
        ):
            if batch is None:
                continue
            features, _targets = batch
            sample_indices_feature = features.pop("__dataset_index__", None)
            if sample_indices_feature is None:
                raise RuntimeError("Missing __dataset_index__ in B2D bank batch.")
            if torch.is_tensor(sample_indices_feature):
                batch_indices = [int(idx) for idx in sample_indices_feature.tolist()]
            else:
                batch_indices = [int(idx) for idx in sample_indices_feature]
            batch_size = _get_batch_size(features)
            if batch_size <= 0:
                continue
            if len(batch_indices) != batch_size:
                raise RuntimeError("Mismatch between filtered B2D bank batch size and dataset indices.")

            batch_tokens = [str(token) for token in features.get("scenario_token", [])]
            batch_log_names = [str(log_name) for log_name in features.get("scenario_log_name", [])]
            if len(batch_tokens) != batch_size or len(batch_log_names) != batch_size:
                raise RuntimeError("Mismatch between batch size and B2D token/log_name counts.")

            features_device = _to_device_features(features, device)
            scene_features = model.encode_scene_features(features_device).detach().cpu().contiguous()
            ego_global_pose = features_device["ego_global_pose"][:, -1].detach().cpu().contiguous()
            if scene_features.shape[0] != batch_size or ego_global_pose.shape[0] != batch_size:
                raise RuntimeError("Model output shape mismatch while building the B2D memory bank.")

            all_tokens.extend(batch_tokens)
            all_log_names.extend(batch_log_names)
            all_timestamps_us.extend([token_to_record[token].timestamp_us for token in batch_tokens])
            all_sample_indices.extend(batch_indices)
            memory_chunks.append(scene_features)
            pose_chunks.append(ego_global_pose)

    if len(memory_chunks) > 0:
        local_memory_tokens = torch.cat(memory_chunks, dim=0).contiguous()
        local_global_poses = torch.cat(pose_chunks, dim=0).contiguous()
    else:
        local_memory_tokens = torch.empty(0, 0, 0, dtype=torch.float32)
        local_global_poses = torch.empty(0, 3, dtype=torch.float64)

    local_payload = {
        "tokens": all_tokens,
        "log_names": all_log_names,
        "timestamps_us": all_timestamps_us,
        "global_poses": local_global_poses,
        "memory_tokens": local_memory_tokens,
        "sample_indices": all_sample_indices,
    }

    output_path = args.output_path
    if output_path is None:
        raise ValueError("--output-path is required.")

    if world_size > 1:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_id = os.environ.get("LSB_JOBID", str(os.getpid()))
        shard_dir = output_path.parent / f".{output_path.stem}.shards.{run_id}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shard_dir / f"rank_{rank:04d}.pt"
        shard_tmp_path = shard_dir / f".rank_{rank:04d}.pt.tmp"
        torch.save(local_payload, shard_tmp_path)
        shard_tmp_path.replace(shard_path)
        logger.info("Rank %d saved shard payload to %s (%d samples).", rank, shard_path, local_memory_tokens.shape[0])

        if rank != 0:
            return

        _wait_for_shard_files(
            shard_dir=shard_dir,
            world_size=world_size,
            timeout_seconds=float(args.shard_wait_timeout_seconds),
        )
        shard_payloads = [
            torch.load(shard_dir / f"rank_{shard_rank:04d}.pt", map_location="cpu")
            for shard_rank in range(world_size)
        ]
        merged_payload = _merge_shard_payloads(shard_payloads)

        for shard_rank in range(world_size):
            rank_shard_path = shard_dir / f"rank_{shard_rank:04d}.pt"
            if rank_shard_path.exists():
                rank_shard_path.unlink()
        if shard_dir.exists():
            shard_dir.rmdir()
    else:
        merged_payload = _merge_shard_payloads([local_payload])

    all_tokens = list(merged_payload["tokens"])
    all_log_names = list(merged_payload["log_names"])
    global_poses = torch.as_tensor(merged_payload["global_poses"], dtype=torch.float64).contiguous()
    memory_tokens = torch.as_tensor(merged_payload["memory_tokens"], dtype=torch.float32).contiguous()

    precomputed_neighbors = {}
    if not args.disable_precomputed_neighbors:
        if query_records is None:
            precomputed_neighbors = _compute_precomputed_neighbors(
                tokens=all_tokens,
                log_names=all_log_names,
                global_poses=global_poses,
                k=int(args.top_k),
                yaw_distance_weight=float(args.yaw_distance_weight),
                chunk_size=int(args.precompute_chunk_size),
            )
        else:
            precomputed_neighbors = _compute_asymmetric_precomputed_neighbors(
                query_tokens=[record.token for record in query_records],
                query_log_names=[record.log_name for record in query_records],
                query_global_poses=torch.tensor(
                    [record.pose_xyh for record in query_records],
                    dtype=torch.float64,
                ),
                bank_tokens=all_tokens,
                bank_log_names=all_log_names,
                bank_global_poses=global_poses,
                k=int(args.top_k),
                yaw_distance_weight=float(args.yaw_distance_weight),
                chunk_size=int(args.precompute_chunk_size),
            )

    payload = {
        "tokens": all_tokens,
        "log_names": all_log_names,
        "global_poses": global_poses,
        "memory_tokens": memory_tokens,
        "train_neighbors_tokens": precomputed_neighbors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    output_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
    logger.info(
        "Saved B2D episodic memory bank to %s with %d entries, tokens_per_memory=%d, dim=%d, size=%.2f MB.",
        output_path,
        memory_tokens.shape[0],
        memory_tokens.shape[1],
        memory_tokens.shape[2],
        output_size_mb,
    )


if __name__ == "__main__":
    main()
