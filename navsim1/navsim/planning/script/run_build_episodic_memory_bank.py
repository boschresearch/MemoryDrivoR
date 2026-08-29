# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
from navsim.agents.drivoR.epi_mem_modules.log_group import normalize_log_group
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _strip_torchrun_local_rank_arg() -> None:
    """Hydra does not accept torchrun's injected --local-rank argument."""
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


def _resolve_bank_log_names(cfg: DictConfig) -> List[str]:
    train_logs = [str(log_name) for log_name in cfg.get("train_logs", [])]
    include_val_logs = bool(cfg.get("include_val_logs_in_bank", False))
    if not include_val_logs:
        return train_logs

    val_logs = [str(log_name) for log_name in cfg.get("val_logs", [])]
    return list(dict.fromkeys(train_logs + val_logs))


def _build_train_dataset(cfg: DictConfig, feature_builders):
    bank_log_names = _resolve_bank_log_names(cfg)
    logger.info(
        "Building episodic bank from %d logs (include_val_logs_in_bank=%s)",
        len(bank_log_names),
        bool(cfg.get("include_val_logs_in_bank", False)),
    )

    if cfg.use_cache_without_dataset:
        return CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=feature_builders,
            target_builders=[],
            log_names=bank_log_names,
        )

    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        bank_log_name_set = set(bank_log_names)
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in bank_log_name_set
        ]
    else:
        train_scene_filter.log_names = bank_log_names

    agent_cfg = cfg.agent.config
    sensor_config = SensorConfig(
        cam_f0=OmegaConf.to_object(agent_cfg["cam_f0"]),
        cam_l0=OmegaConf.to_object(agent_cfg["cam_l0"]),
        cam_l1=OmegaConf.to_object(agent_cfg["cam_l1"]),
        cam_l2=OmegaConf.to_object(agent_cfg["cam_l2"]),
        cam_r0=OmegaConf.to_object(agent_cfg["cam_r0"]),
        cam_r1=OmegaConf.to_object(agent_cfg["cam_r1"]),
        cam_r2=OmegaConf.to_object(agent_cfg["cam_r2"]),
        cam_b0=OmegaConf.to_object(agent_cfg["cam_b0"]),
        lidar_pc=OmegaConf.to_object(agent_cfg["lidar_pc"]),
    )

    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=train_scene_filter,
        sensor_config=sensor_config,
    )
    return Dataset(
        scene_loader=scene_loader,
        feature_builders=feature_builders,
        target_builders=[],
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=False,
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
    if len(tokens) != n_samples:
        raise ValueError(
            f"`tokens` length ({len(tokens)}) must match global_poses length ({n_samples})."
        )
    if len(log_names) != n_samples:
        raise ValueError(
            f"`log_names` length ({len(log_names)}) must match global_poses length ({n_samples})."
        )
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
    # Use an enlarged candidate pool because we later enforce strict cross-log
    # filtering and one-neighbor-per-log diversity.
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
            row = ranked_idx[row_idx].tolist()

            for cand_idx in row:
                cand_idx = int(cand_idx)
                if cand_idx == self_global_idx:
                    continue
                cand_log_id = log_ids[cand_idx]
                if cand_log_id == query_log_id:
                    continue
                if cand_log_id in used_neighbor_log_ids:
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
            "%d/%d samples had fewer than %d precomputed cross-log-group neighbors (k=%d, unique_log_groups=%d, cand_k=%d).",
            underfilled_count,
            n_samples,
            target_k,
            k,
            len(unique_log_groups),
            cand_k,
        )

    return neighbors


def _load_drivor_checkpoint_if_provided(model: torch.nn.Module, checkpoint_path: str) -> None:
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
    logger.info("Checkpoint load: missing=%d unexpected=%d", len(incompatible.missing_keys), len(incompatible.unexpected_keys))


def _merge_shard_payloads(shard_payloads: List[Dict]) -> Dict:
    merge_start = time.perf_counter()
    logger.info("Merging %d shard payload(s) into a single episodic memory bank payload.", len(shard_payloads))

    all_indices: List[int] = []
    all_tokens: List[str] = []
    all_log_names: List[str] = []
    memory_parts: List[torch.Tensor] = []
    pose_parts: List[torch.Tensor] = []

    validate_start = time.perf_counter()
    for shard_payload in tqdm(
        shard_payloads,
        desc="Validating shard payloads",
        disable=(len(shard_payloads) <= 1),
    ):
        shard_tokens = [str(token) for token in shard_payload.get("tokens", [])]
        shard_indices = [int(idx) for idx in shard_payload.get("sample_indices", [])]
        if len(shard_tokens) != len(shard_indices):
            raise ValueError("Distributed shard payload has inconsistent token/index lengths.")
        if len(shard_tokens) == 0:
            continue

        shard_log_names = [str(log_name) for log_name in shard_payload.get("log_names", [])]
        if len(shard_log_names) != len(shard_tokens):
            shard_log_names = [""] * len(shard_tokens)

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
        memory_parts.append(shard_memory)
        pose_parts.append(shard_poses)
    logger.info(
        "Shard payload validation complete in %.2fs (kept %d non-empty shard payload(s)).",
        time.perf_counter() - validate_start,
        len(memory_parts),
    )

    if len(memory_parts) == 0:
        raise RuntimeError("No samples extracted while building episodic memory bank.")
    if len(set(all_indices)) != len(all_indices):
        raise RuntimeError("Distributed shard merge found duplicate sample indices.")

    cat_start = time.perf_counter()
    memory_tokens = torch.cat(memory_parts, dim=0).contiguous()
    global_poses = torch.cat(pose_parts, dim=0).contiguous()
    logger.info(
        "Concatenated shard tensors in %.2fs (memory_tokens=%s, global_poses=%s).",
        time.perf_counter() - cat_start,
        tuple(memory_tokens.shape),
        tuple(global_poses.shape),
    )

    sort_start = time.perf_counter()
    sort_order = torch.argsort(torch.as_tensor(all_indices, dtype=torch.long))
    sort_order_list = sort_order.tolist()
    memory_tokens = memory_tokens[sort_order]
    global_poses = global_poses[sort_order]
    logger.info("Sorted merged payload by sample index in %.2fs.", time.perf_counter() - sort_start)
    logger.info("Shard merge total time: %.2fs.", time.perf_counter() - merge_start)

    return {
        "tokens": [all_tokens[idx] for idx in sort_order_list],
        "log_names": [all_log_names[idx] for idx in sort_order_list],
        "global_poses": global_poses,
        "memory_tokens": memory_tokens,
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


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    rank, world_size, local_rank = 0, 1, 0
    try:
        rank, world_size, local_rank = _init_distributed_context()
        agent_cfg = cfg.agent.config

        if "epi_memory" not in agent_cfg or not agent_cfg.epi_memory.get("enabled", False):
            raise ValueError("Enable `agent.config.epi_memory.enabled=true` before building the memory bank.")

        bank_cfg = agent_cfg.epi_memory.bank
        output_path = bank_cfg.get("path", None)
        if not output_path:
            raise ValueError("Set `agent.config.epi_memory.bank.path` to an output `.pt` path.")
        output_path = Path(output_path)

        logger.info(
            "Starting episodic memory extraction on rank=%d, local_rank=%d, world_size=%d",
            rank,
            local_rank,
            world_size,
        )

        logger.info("Building DrivoR model for episodic memory bank extraction")
        # Build a model without trying to load an episodic bank (we're about to create it).
        # Keep the rest of the config (e.g. frozen backbone) unchanged.
        model_cfg = OmegaConf.create(OmegaConf.to_container(agent_cfg, resolve=True))
        if "epi_memory" in model_cfg and model_cfg.epi_memory is not None and "bank" in model_cfg.epi_memory:
            model_cfg.epi_memory.bank.path = None

        model = DrivoRModel(model_cfg)
        _load_drivor_checkpoint_if_provided(model, str(cfg.agent.get("checkpoint_path", "")))

        feature_builders = [DrivoRFeatureBuilder(config=agent_cfg)]
        full_dataset = _build_train_dataset(cfg, feature_builders)

        if world_size > 1:
            rank_indices = list(range(rank, len(full_dataset), world_size))
            dataset = Subset(full_dataset, rank_indices)
        else:
            rank_indices = list(range(len(full_dataset)))
            dataset = full_dataset
        loader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False, drop_last=False)

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
        all_sample_indices: List[int] = []
        memory_chunks: List[torch.Tensor] = []
        pose_chunks: List[torch.Tensor] = []
        consumed_rank_indices = 0

        with torch.no_grad():
            for features, _targets in tqdm(
                loader,
                desc=f"Extracting scene features (rank {rank})",
                disable=(rank != 0),
            ):
                batch_size = _get_batch_size(features)
                if batch_size <= 0:
                    continue
                batch_indices = rank_indices[consumed_rank_indices : consumed_rank_indices + batch_size]
                consumed_rank_indices += batch_size
                if len(batch_indices) != batch_size:
                    raise RuntimeError("Failed to map batch to dataset indices during distributed extraction.")

                batch_tokens = [str(token) for token in features.get("scenario_token", [])]
                if len(batch_tokens) == 0:
                    continue
                if len(batch_tokens) != batch_size:
                    raise RuntimeError(
                        f"Mismatch between batch size ({batch_size}) and scenario_token count ({len(batch_tokens)})."
                    )
                batch_log_names = [str(log_name) for log_name in features.get("scenario_log_name", [])]

                features_device = _to_device_features(features, device)
                if "ego_global_pose" not in features_device:
                    raise KeyError(
                        "Missing `ego_global_pose` in cached features. "
                        "Re-run dataset caching after updating DrivoRFeatureBuilder."
                    )
                scene_features = model.encode_scene_features(features_device).detach().cpu().contiguous()
                ego_global_pose = features_device["ego_global_pose"][:, -1].detach().cpu().contiguous()
                if scene_features.shape[0] != batch_size or ego_global_pose.shape[0] != batch_size:
                    raise RuntimeError("Model output shape mismatch while building episodic memory bank.")

                all_tokens.extend(batch_tokens)
                if len(batch_log_names) == len(batch_tokens):
                    all_log_names.extend(batch_log_names)
                else:
                    all_log_names.extend([""] * len(batch_tokens))
                all_sample_indices.extend(batch_indices)
                memory_chunks.append(scene_features)
                pose_chunks.append(ego_global_pose)

        if consumed_rank_indices != len(rank_indices):
            raise RuntimeError(
                f"Consumed {consumed_rank_indices} samples but expected {len(rank_indices)} for rank {rank}."
            )

        if len(memory_chunks) > 0:
            local_memory_tokens = torch.cat(memory_chunks, dim=0).contiguous()
            local_global_poses = torch.cat(pose_chunks, dim=0).contiguous()
        else:
            local_memory_tokens = torch.empty(0, 0, 0, dtype=torch.float32)
            local_global_poses = torch.empty(0, 3, dtype=torch.float64)

        if local_memory_tokens.shape[0] != len(all_tokens) or local_memory_tokens.shape[0] != len(all_sample_indices):
            raise RuntimeError("Local episodic memory payload has inconsistent tensor/list lengths.")

        local_payload = {
            "tokens": all_tokens,
            "log_names": all_log_names,
            "global_poses": local_global_poses,
            "memory_tokens": local_memory_tokens,
            "sample_indices": all_sample_indices,
        }

        if world_size > 1:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            run_id = os.environ.get("LSB_JOBID", str(os.getpid()))
            shard_dir = output_path.parent / f".{output_path.stem}.shards.{run_id}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / f"rank_{rank:04d}.pt"
            shard_tmp_path = shard_dir / f".rank_{rank:04d}.pt.tmp"
            shard_save_start = time.perf_counter()
            torch.save(local_payload, shard_tmp_path)
            shard_tmp_path.replace(shard_path)
            logger.info(
                "Rank %d saved shard payload to %s in %.2fs (%d samples).",
                rank,
                shard_path,
                time.perf_counter() - shard_save_start,
                local_memory_tokens.shape[0],
            )

            if rank != 0:
                return

            shard_wait_timeout_seconds = float(bank_cfg.get("shard_wait_timeout_seconds", 7200.0))
            wait_start = time.perf_counter()
            logger.info(
                "Rank 0 waiting for %d shard file(s) before merge (timeout=%.1fs).",
                world_size,
                shard_wait_timeout_seconds,
            )
            _wait_for_shard_files(
                shard_dir=shard_dir,
                world_size=world_size,
                timeout_seconds=shard_wait_timeout_seconds,
            )
            logger.info("Shard file wait completed in %.2fs.", time.perf_counter() - wait_start)

            shard_payloads: List[Dict] = []
            load_start = time.perf_counter()
            for shard_rank in tqdm(range(world_size), desc="Loading shard files"):
                rank_shard_path = shard_dir / f"rank_{shard_rank:04d}.pt"
                if not rank_shard_path.exists():
                    raise FileNotFoundError(f"Missing shard file during merge: {rank_shard_path}")
                shard_payloads.append(torch.load(rank_shard_path, map_location="cpu"))
            logger.info("Loaded %d shard file(s) in %.2fs.", len(shard_payloads), time.perf_counter() - load_start)

            merge_start = time.perf_counter()
            merged_payload = _merge_shard_payloads(shard_payloads)
            logger.info("Finished shard payload merge call in %.2fs.", time.perf_counter() - merge_start)

            cleanup_start = time.perf_counter()
            for shard_rank in tqdm(range(world_size), desc="Cleaning shard files"):
                rank_shard_path = shard_dir / f"rank_{shard_rank:04d}.pt"
                if rank_shard_path.exists():
                    rank_shard_path.unlink()
            if shard_dir.exists():
                shard_dir.rmdir()
            logger.info("Cleaned shard artifacts in %.2fs.", time.perf_counter() - cleanup_start)
        else:
            logger.info("Single-rank run; merging local payload directly.")
            merged_payload = _merge_shard_payloads([local_payload])

        materialize_start = time.perf_counter()
        all_tokens = list(merged_payload["tokens"])
        all_log_names = list(merged_payload["log_names"])
        global_poses = torch.as_tensor(merged_payload["global_poses"], dtype=torch.float64).contiguous()
        memory_tokens = torch.as_tensor(merged_payload["memory_tokens"], dtype=torch.float32).contiguous()
        logger.info(
            "Materialized merged payload tensors/lists in %.2fs (entries=%d).",
            time.perf_counter() - materialize_start,
            memory_tokens.shape[0],
        )

        precomputed_neighbors = {}
        if bank_cfg.get("use_precomputed_train_neighbors", True):
            precompute_start = time.perf_counter()
            logger.info(
                "Starting precomputed neighbor build (k=%d, yaw_distance_weight=%.4f, chunk_size=%d).",
                int(bank_cfg.get("top_k", 0)),
                float(bank_cfg.get("yaw_distance_weight", 0.0)),
                int(bank_cfg.get("precompute_chunk_size", 1024)),
            )
            precomputed_neighbors = _compute_precomputed_neighbors(
                tokens=all_tokens,
                log_names=all_log_names,
                global_poses=global_poses,
                k=int(bank_cfg.get("top_k", 0)),
                yaw_distance_weight=float(bank_cfg.get("yaw_distance_weight", 0.0)),
                chunk_size=int(bank_cfg.get("precompute_chunk_size", 1024)),
            )
            logger.info(
                "Finished precomputed neighbor build in %.2fs (token_count=%d).",
                time.perf_counter() - precompute_start,
                len(precomputed_neighbors),
            )

        payload = {
            "tokens": all_tokens,
            "log_names": all_log_names,
            "global_poses": global_poses,
            "memory_tokens": memory_tokens,
            "train_neighbors_tokens": precomputed_neighbors,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_start = time.perf_counter()
        torch.save(payload, output_path)
        output_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
        logger.info("Final payload torch.save completed in %.2fs (size=%.2f MB).", time.perf_counter() - save_start, output_size_mb)
        logger.info(
            "Saved episodic memory bank to %s with %d entries, tokens_per_memory=%d, dim=%d",
            output_path,
            memory_tokens.shape[0],
            memory_tokens.shape[1],
            memory_tokens.shape[2],
        )
    finally:
        # No process-group teardown is needed here because this script does not
        # initialize torch.distributed for synchronization.
        pass


if __name__ == "__main__":
    main()
