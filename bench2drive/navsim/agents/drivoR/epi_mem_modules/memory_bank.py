# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import pickle

import torch

from .log_group import extract_town_name, normalize_log_group


def _pose_to_matrix(pose_xyh: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pose_xyh: (..., 3) in global coordinates [x, y, heading].
    Returns:
        (..., 4, 4) homogeneous transform.
    """
    c = torch.cos(pose_xyh[..., 2])
    s = torch.sin(pose_xyh[..., 2])

    out = torch.zeros(*pose_xyh.shape[:-1], 4, 4, dtype=pose_xyh.dtype, device=pose_xyh.device)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., 2, 2] = 1.0
    out[..., 3, 3] = 1.0
    out[..., 0, 3] = pose_xyh[..., 0]
    out[..., 1, 3] = pose_xyh[..., 1]
    return out


class EpisodicMemoryBank:
    """
    Episodic memory retrieval utility for DrivoR.

    Expected bank file keys:
      - `memory_tokens`: Tensor [N, T, D]
      - `global_poses`: Tensor [N, 3]
      - `tokens`: List[str]
    Optional keys:
      - `log_names`: List[str]
      - `train_neighbors_tokens`: Dict[str, List[str]]
      - `train_neighbors`: Dict[str, List[int]]
    """

    def __init__(
        self,
        bank_path: Optional[str],
        top_k: int,
        use_precomputed_train_neighbors: bool = True,
        yaw_distance_weight: float = 0.0,
        exclude_self: bool = True,
        max_distance_m: Optional[float] = None,
        min_yaw: Optional[float] = None,
        max_yaw: Optional[float] = None,
    ):
        self.top_k = max(int(top_k), 0)
        self.use_precomputed_train_neighbors = use_precomputed_train_neighbors
        self.yaw_distance_weight = float(yaw_distance_weight)
        self.exclude_self = exclude_self
        max_distance = None if max_distance_m is None else float(max_distance_m)
        self.max_distance_m = max_distance if max_distance is not None and max_distance > 0 else None
        min_yaw_val = None if min_yaw is None else float(min_yaw)
        max_yaw_val = None if max_yaw is None else float(max_yaw)
        self.min_yaw = min_yaw_val if min_yaw_val is not None and min_yaw_val >= 0 else None
        self.max_yaw = max_yaw_val if max_yaw_val is not None and max_yaw_val > 0 else None
        if self.min_yaw is not None and self.max_yaw is not None and self.max_yaw < self.min_yaw:
            raise ValueError(
                f"Invalid yaw range: max_yaw ({self.max_yaw}) must be >= min_yaw ({self.min_yaw})."
            )

        self.tokens: List[str] = []
        self.token_to_idx: Dict[str, int] = {}
        self.log_names: Optional[List[str]] = None
        self.log_groups: Optional[List[str]] = None
        self.town_names: Optional[List[str]] = None
        self.memory_tokens: Optional[torch.Tensor] = None
        self.global_poses: Optional[torch.Tensor] = None
        self.train_neighbors_tokens: Dict[str, List[str]] = {}
        # Optional fast path for online KNN (built lazily to keep scipy optional).
        self._kdtree = None
        self._kdtree_xy = None

        if bank_path:
            self._load(Path(bank_path))

    @property
    def dim(self) -> int:
        if self.memory_tokens is None:
            return 0
        return int(self.memory_tokens.shape[-1])

    @property
    def tokens_per_memory(self) -> int:
        if self.memory_tokens is None:
            return 0
        return int(self.memory_tokens.shape[1])

    def _load(self, bank_path: Path) -> None:
        if not bank_path.exists():
            raise FileNotFoundError(f"Episodic memory bank not found: {bank_path}")

        if bank_path.suffix in {".pt", ".pth"}:
            bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        else:
            with open(bank_path, "rb") as f:
                bank = pickle.load(f)

        required = {"memory_tokens", "global_poses", "tokens"}
        missing = required - set(bank.keys())
        if missing:
            raise KeyError(f"Missing keys in episodic memory bank: {sorted(missing)}")

        self.memory_tokens = torch.as_tensor(bank["memory_tokens"], dtype=torch.float32).contiguous()
        self.global_poses = torch.as_tensor(bank["global_poses"], dtype=torch.float64).contiguous()
        self.tokens = list(bank["tokens"])
        self.log_names = list(bank["log_names"]) if "log_names" in bank else None
        self.log_groups = (
            [normalize_log_group(log_name) for log_name in self.log_names]
            if self.log_names is not None
            else None
        )
        self.town_names = (
            [extract_town_name(log_name) for log_name in self.log_names]
            if self.log_names is not None
            else None
        )

        if self.memory_tokens.ndim != 3:
            raise ValueError(f"memory_tokens must be [N, T, D], got {tuple(self.memory_tokens.shape)}")
        if self.global_poses.shape != (self.memory_tokens.shape[0], 3):
            raise ValueError(
                "global_poses must match memory_tokens length: "
                f"{tuple(self.global_poses.shape)} vs {tuple(self.memory_tokens.shape)}"
            )
        if len(self.tokens) != self.memory_tokens.shape[0]:
            raise ValueError(
                f"tokens length {len(self.tokens)} does not match memory bank size {self.memory_tokens.shape[0]}"
            )

        if "train_neighbors_tokens" in bank:
            self.train_neighbors_tokens = {
                str(k): [str(vv) for vv in v] for k, v in bank["train_neighbors_tokens"].items()
            }
        elif "train_neighbors" in bank:
            # Backward compatibility with index-based caches.
            self.train_neighbors_tokens = {}
            for token, idxs in bank["train_neighbors"].items():
                token_key = str(token)
                mapped_tokens = []
                for idx in idxs:
                    if 0 <= int(idx) < len(self.tokens):
                        mapped_tokens.append(self.tokens[int(idx)])
                self.train_neighbors_tokens[token_key] = mapped_tokens

        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        assert self.memory_tokens.shape[0] > 0, "Episodic memory bank is empty after loading."

    def _to_optional_str_list(
        self, values: Optional[Union[Sequence[str], torch.Tensor, str]], batch_size: int
    ) -> List[Optional[str]]:
        if values is None:
            return [None] * batch_size
        if isinstance(values, str):
            # Common single-sample case (avoid iterating over characters).
            return [values] + [None] * (batch_size - 1)
        if isinstance(values, torch.Tensor):
            # Not expected, but keep behavior deterministic.
            values = [str(v.item()) for v in values]
        out = [None if value is None else str(value) for value in values]
        if len(out) < batch_size:
            out = out + [None] * (batch_size - len(out))
        return out[:batch_size]

    def _to_token_list(
        self, scenario_tokens: Optional[Union[Sequence[str], torch.Tensor]], batch_size: int
    ) -> List[Optional[str]]:
        return self._to_optional_str_list(scenario_tokens, batch_size)

    def _ensure_kdtree(self) -> None:
        if self._kdtree is not None:
            return
        if self.global_poses is None or self.global_poses.numel() == 0:
            return

        try:
            from scipy.spatial import cKDTree  # type: ignore
        except Exception:
            return

        xy = self.global_poses[:, :2].detach().cpu().numpy()
        self._kdtree_xy = xy
        self._kdtree = cKDTree(xy)

    def _candidate_pool_size(self, bank_size: int) -> int:
        cand_k = self.top_k + (1 if self.exclude_self else 0)
        if self.log_groups is not None:
            # Enlarge candidate pool to satisfy one-neighbor-per-source-log-group constraint.
            cand_k = max(cand_k, 64, self.top_k * 32)
        if self.yaw_distance_weight > 0:
            cand_k = max(cand_k, 32, self.top_k * 10)
            if self.log_groups is not None:
                cand_k = max(cand_k, 128, self.top_k * 64)
        return min(cand_k, bank_size)

    def _select_log_diverse_indices(
        self,
        ranked_indices: Sequence[int],
        scenario_token: Optional[str],
        scenario_log_name: Optional[str] = None,
        scenario_town_name: Optional[str] = None,
    ) -> List[int]:
        if self.top_k <= 0:
            return []

        scenario_idx = self.token_to_idx.get(scenario_token) if scenario_token is not None else None
        scenario_log_group: Optional[str] = normalize_log_group(scenario_log_name) if scenario_log_name else None
        resolved_scenario_town_name: str = str(scenario_town_name) if scenario_town_name else ""
        if not resolved_scenario_town_name and scenario_log_name:
            resolved_scenario_town_name = extract_town_name(scenario_log_name)
        if (
            scenario_idx is not None
            and self.log_groups is not None
            and 0 <= scenario_idx < len(self.log_groups)
        ):
            scenario_log_group = self.log_groups[scenario_idx]
        if (
            scenario_idx is not None
            and self.town_names is not None
            and 0 <= scenario_idx < len(self.town_names)
            and self.town_names[scenario_idx]
        ):
            resolved_scenario_town_name = self.town_names[scenario_idx]

        out: List[int] = []
        used_neighbor_logs = set()
        for idx in ranked_indices:
            idx = int(idx)
            if idx < 0 or idx >= len(self.tokens):
                continue
            if self.exclude_self and scenario_idx is not None and idx == scenario_idx:
                continue

            if self.town_names is not None and resolved_scenario_town_name:
                cand_town_name = self.town_names[idx]
                if cand_town_name != resolved_scenario_town_name:
                    continue

            if self.log_groups is not None:
                cand_log_name = self.log_groups[idx]
                # Empty/unknown log groups do not participate in diversity filtering.
                if cand_log_name:
                    if scenario_log_group is not None and cand_log_name == scenario_log_group:
                        continue
                    if cand_log_name in used_neighbor_logs:
                        continue
                    used_neighbor_logs.add(cand_log_name)

            out.append(idx)
            if len(out) >= self.top_k:
                break
        return out

    def _online_knn_indices(
        self,
        query_poses: torch.Tensor,
        scenario_tokens: List[Optional[str]],
        scenario_log_names: List[Optional[str]],
        scenario_town_names: List[Optional[str]],
    ) -> List[List[int]]:
        assert self.global_poses is not None
        query_poses_cpu = query_poses[:, :3].detach().cpu().to(torch.float64)
        bank_poses = self.global_poses
        bank_size = int(bank_poses.shape[0])
        if bank_size == 0:
            return [[] for _ in range(query_poses_cpu.shape[0])]

        self._ensure_kdtree()
        if self._kdtree is not None:
            cand_k = self._candidate_pool_size(bank_size)

            _, idx = self._kdtree.query(query_poses_cpu[:, :2].numpy(), k=cand_k)
            if cand_k == 1:
                idx = idx[:, None]

            idx_t = torch.as_tensor(idx, dtype=torch.long)
            if self.yaw_distance_weight > 0 and cand_k > 0:
                q_xy = query_poses_cpu[:, None, :2]
                cand_xy = bank_poses[idx_t, :2]
                d_xy = (q_xy - cand_xy).square().sum(dim=-1)

                q_yaw = query_poses_cpu[:, None, 2]
                cand_yaw = bank_poses[idx_t, 2]
                yaw_delta = q_yaw - cand_yaw
                yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
                d_all = d_xy + self.yaw_distance_weight * yaw_delta.square()
                rank_order = torch.argsort(d_all, dim=-1)
                ranked_idx = idx_t.gather(1, rank_order)

                indices: List[List[int]] = []
                for b in range(query_poses_cpu.shape[0]):
                    indices.append(
                        self._select_log_diverse_indices(
                            ranked_indices=ranked_idx[b].tolist(),
                            scenario_token=scenario_tokens[b],
                            scenario_log_name=scenario_log_names[b],
                            scenario_town_name=scenario_town_names[b],
                        )
                    )
                return indices

            indices: List[List[int]] = []
            for b in range(query_poses_cpu.shape[0]):
                row = idx_t[b].tolist()
                indices.append(
                    self._select_log_diverse_indices(
                        row,
                        scenario_tokens[b],
                        scenario_log_names[b],
                        scenario_town_names[b],
                    )
                )
            return indices

        # Fallback: brute-force over the full bank.
        d_xy = torch.cdist(query_poses_cpu[:, :2], bank_poses[:, :2], p=2).square()
        d_all = d_xy
        if self.yaw_distance_weight > 0:
            yaw_delta = query_poses_cpu[:, None, 2] - bank_poses[None, :, 2]
            yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta))
            d_all = d_all + self.yaw_distance_weight * yaw_delta.square()

        indices: List[List[int]] = []
        k = self._candidate_pool_size(bank_size)
        for batch_idx in range(query_poses_cpu.shape[0]):
            row = d_all[batch_idx]
            token = scenario_tokens[batch_idx]
            if self.exclude_self and token is not None and token in self.token_to_idx:
                row = row.clone()
                row[self.token_to_idx[token]] = float("inf")

            if k == 0:
                indices.append([])
            else:
                ranked = torch.topk(row, k=k, largest=False).indices.tolist()
                chosen = self._select_log_diverse_indices(
                    ranked,
                    token,
                    scenario_log_names[batch_idx],
                    scenario_town_names[batch_idx],
                )
                if len(chosen) < self.top_k and k < bank_size and self.log_names is not None:
                    # Expand to full ranking only when the candidate pool was insufficient.
                    full_ranked = torch.argsort(row).tolist()
                    chosen = self._select_log_diverse_indices(
                        full_ranked,
                        token,
                        scenario_log_names[batch_idx],
                        scenario_town_names[batch_idx],
                    )
                indices.append(chosen)
        return indices

    def _precomputed_indices(
        self,
        token: Optional[str],
        scenario_log_name: Optional[str] = None,
        scenario_town_name: Optional[str] = None,
    ) -> List[int]:
        if token is None or token not in self.train_neighbors_tokens:
            return []
        neighbors = self.train_neighbors_tokens[token]
        ranked: List[int] = []
        for neighbor_token in neighbors:
            if neighbor_token in self.token_to_idx:
                ranked.append(self.token_to_idx[neighbor_token])
            if len(ranked) >= self.top_k * 8:
                break
        return self._select_log_diverse_indices(ranked, token, scenario_log_name, scenario_town_name)

    def query(
        self,
        query_global_pose: torch.Tensor,
        scenario_tokens: Optional[Union[Sequence[str], torch.Tensor]] = None,
        scenario_log_names: Optional[Sequence[str]] = None,
        scenario_town_names: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns:
          - episodic memory tokens per batch sample: list[[Ki, T, D]]
          - relative transforms memory->current per sample: list[[Ki, 4, 4]]
          where Ki is variable (0 <= Ki <= top_k) for each sample.
        """
        if query_global_pose.ndim == 3:
            query_global_pose = query_global_pose[:, -1]
        if query_global_pose.ndim != 2 or query_global_pose.shape[-1] < 3:
            raise ValueError(
                f"query_global_pose must be [B, 3] or [B, H, 3], got {tuple(query_global_pose.shape)}"
            )

        if device is None:
            device = query_global_pose.device

        batch_size = query_global_pose.shape[0]
        if self.top_k == 0:
            empty_mem = torch.empty(0, 0, 0, device=device)
            empty_pose = torch.empty(0, 4, 4, device=device)
            return [empty_mem for _ in range(batch_size)], [empty_pose for _ in range(batch_size)]

        assert self.memory_tokens is not None and self.global_poses is not None
        token_list = self._to_token_list(scenario_tokens, batch_size)
        log_name_list = self._to_optional_str_list(scenario_log_names, batch_size)
        town_name_list = self._to_optional_str_list(scenario_town_names, batch_size)

        index_lists: List[List[int]] = []
        if self.use_precomputed_train_neighbors:
            for batch_idx, token in enumerate(token_list):
                index_lists.append(
                    self._precomputed_indices(
                        token,
                        log_name_list[batch_idx],
                        town_name_list[batch_idx],
                    )
                )
        else:
            index_lists = self._online_knn_indices(
                query_global_pose,
                token_list,
                log_name_list,
                town_name_list,
            )

        query_pose_cpu = query_global_pose[:, :3].detach().cpu().to(torch.float64)
        query_tf = _pose_to_matrix(query_pose_cpu)
        query_tf_inv = torch.linalg.inv(query_tf)

        memory_list: List[torch.Tensor] = []
        rel_pose_list: List[torch.Tensor] = []
        max_distance_sq = (
            float(self.max_distance_m) * float(self.max_distance_m)
            if self.max_distance_m is not None
            else None
        )
        use_yaw_filter = self.min_yaw is not None or self.max_yaw is not None
        for batch_idx in range(batch_size):
            idxs = list(index_lists[batch_idx])
            if max_distance_sq is not None and len(idxs) > 0:
                q_xy = query_pose_cpu[batch_idx, :2]
                cand_xy = self.global_poses[idxs, :2]
                d_sq = (cand_xy - q_xy.unsqueeze(0)).square().sum(dim=-1)
                keep = d_sq <= max_distance_sq
                idxs = [idx for idx, is_valid in zip(idxs, keep.tolist()) if is_valid]
            if use_yaw_filter and len(idxs) > 0:
                q_yaw = query_pose_cpu[batch_idx, 2]
                cand_yaw = self.global_poses[idxs, 2]
                yaw_delta = q_yaw - cand_yaw
                yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)).abs()
                keep = torch.ones_like(yaw_delta, dtype=torch.bool)
                if self.min_yaw is not None:
                    keep &= yaw_delta >= float(self.min_yaw)
                if self.max_yaw is not None:
                    keep &= yaw_delta <= float(self.max_yaw)
                idxs = [idx for idx, is_valid in zip(idxs, keep.tolist()) if is_valid]

            if len(idxs) == 0:
                mem = torch.empty(0, self.tokens_per_memory, self.dim, dtype=torch.float32)
                rel = torch.empty(0, 4, 4, dtype=torch.float32)
            else:
                idxs = idxs[: self.top_k]
                mem = self.memory_tokens[idxs].clone()
                mem_pose = self.global_poses[idxs]
                mem_tf = _pose_to_matrix(mem_pose)
                rel = (query_tf_inv[batch_idx].unsqueeze(0) @ mem_tf).to(torch.float32)

            memory_list.append(mem.to(device=device, non_blocking=True))
            rel_pose_list.append(rel.to(device=device, non_blocking=True))

        return memory_list, rel_pose_list
