# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import List, Optional, Union

import torch
from torch import nn

from .perceiver_resampler import PerceiverResampler
from .pose_embedder import PoseEmbedder


class EpiMemoryInjector(nn.Module):
    """Append resampled episodic-memory tokens to the current scene tokens."""

    def __init__(
        self,
        resampler: Optional[dict] = None,
        pose_embedder: Optional[dict] = None,
        output_projector: Optional[nn.Module] = None,
        use_output_proj: bool = True,
        use_epi_flag: bool = True,
        add_pose_before_resamp: bool = True,
        dim: int = 256,
        out_dim: Optional[int] = None,
        mask_current_tokens_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.use_output_proj = use_output_proj
        self.use_epi_flag = use_epi_flag
        self.add_pose_before_resamp = add_pose_before_resamp
        self.mask_current_tokens_prob = float(mask_current_tokens_prob)
        if not 0.0 <= self.mask_current_tokens_prob <= 1.0:
            raise ValueError(
                f"Invalid mask_current_tokens_prob={self.mask_current_tokens_prob}. Expected [0, 1]."
            )

        if self.use_output_proj:
            self.output_projector = output_projector or nn.Linear(dim, self.out_dim)
            embedding_dim = self.out_dim
        else:
            self.output_projector = nn.Identity()
            embedding_dim = self.dim

        if self.use_epi_flag:
            self.curr_flag_emb = nn.Parameter(torch.empty(1, embedding_dim))
            self.epi_flag_emb = nn.Parameter(torch.empty(1, embedding_dim))
            nn.init.normal_(self.curr_flag_emb, std=0.02)
            nn.init.normal_(self.epi_flag_emb, std=0.02)

        self.resampler: Optional[PerceiverResampler] = self._build_module(
            resampler, PerceiverResampler
        )
        self.pose_embedder: Optional[PoseEmbedder] = self._build_module(
            pose_embedder, PoseEmbedder
        )

    def _build_module(self, cfg, module_cls):
        if cfg is None:
            return None
        if isinstance(cfg, nn.Module):
            return cfg
        if not isinstance(cfg, dict):
            raise TypeError(f"Invalid config type for {module_cls.__name__}: {type(cfg)}")

        module_cfg = dict(cfg)
        module_cfg["dim"] = self.dim
        return module_cls(**module_cfg)

    def forward(
        self,
        current_embed: torch.Tensor,
        epi_tokens: List[torch.Tensor],
        epi_poses: Optional[List[torch.Tensor]] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if self.use_epi_flag:
            current_embed = current_embed + self.curr_flag_emb
        if len(epi_tokens) == 0:
            return current_embed

        current_tokens = self._mask_current_tokens(current_embed, epi_tokens)
        resampled_memories = self._resample_memories(epi_tokens, epi_poses)
        if isinstance(current_tokens, torch.Tensor):
            current_tokens = list(current_tokens)

        planner_input = [
            torch.cat([current, memory])
            for current, memory in zip(current_tokens, resampled_memories)
        ]
        sequence_lengths = [sample.shape[0] for sample in planner_input]
        if len(set(sequence_lengths)) == 1:
            return torch.stack(planner_input, dim=0)
        return planner_input

    def _mask_current_tokens(
        self,
        current_embed: torch.Tensor,
        epi_tokens: List[torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        batch_size, num_tokens = current_embed.shape[:2]
        if len(epi_tokens) != batch_size:
            raise ValueError(
                f"Expected epi_tokens length to match batch size ({batch_size}), got {len(epi_tokens)}."
            )

        memory_available = torch.tensor(
            [len(sample) > 0 for sample in epi_tokens], device=current_embed.device
        )
        if (
            num_tokens == 0
            or self.mask_current_tokens_prob <= 0.0
            or not memory_available.any()
            or not self.training
        ):
            return current_embed

        drop_mask = torch.rand(batch_size, device=current_embed.device) < self.mask_current_tokens_prob
        drop_mask &= memory_available
        if not drop_mask.any():
            return current_embed
        return [
            current[:0] if is_masked else current
            for current, is_masked in zip(current_embed, drop_mask.tolist())
        ]

    def _resample_memories(
        self,
        epi_tokens: List[torch.Tensor],
        epi_poses: Optional[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        if all(memory.shape[0] == 0 for memory in epi_tokens):
            output_dtype = (
                self.output_projector.weight.dtype
                if isinstance(self.output_projector, nn.Linear)
                else epi_tokens[0].dtype
            )
            return [
                memory.new_zeros((0, self.out_dim), dtype=output_dtype)
                for memory in epi_tokens
            ]

        batch_lengths = [memory.shape[0] for memory in epi_tokens]
        flat_memories = torch.cat(epi_tokens)
        if self.pose_embedder is not None and epi_poses is not None:
            pose_features = self.pose_embedder(torch.cat(epi_poses))[:, None]
        else:
            pose_features = 0

        if self.add_pose_before_resamp:
            flat_memories = flat_memories + pose_features
        if self.resampler is not None:
            flat_memories = self.resampler(flat_memories)
        if not self.add_pose_before_resamp:
            flat_memories = flat_memories + pose_features

        flat_memories = self.output_projector(flat_memories)
        if self.use_epi_flag:
            flat_memories = flat_memories + self.epi_flag_emb

        split_lengths = [length * flat_memories.shape[1] for length in batch_lengths]
        return list(torch.split(flat_memories.flatten(0, 1), split_lengths, dim=0))
