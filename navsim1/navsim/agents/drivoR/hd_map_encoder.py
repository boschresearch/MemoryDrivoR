# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hd_map_schema import (
    HD_MAP_ELEMENT_TYPE_TO_ID,
    HD_MAP_GEOMETRY_TYPE_TO_ID,
    HD_MAP_STOP_LINE_SUBTYPE_UNKNOWN,
)


class RandomFourierPointEmbedding(nn.Module):
    """Random Fourier features for 2D map coordinates, matching PriorDrive's core idea."""

    def __init__(self, embed_dim: int, scale: float = 1.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"`embed_dim` must be even for Fourier features, got {embed_dim}.")

        gaussian_matrix = scale * torch.randn(2, embed_dim // 2)
        self.register_buffer("gaussian_matrix", gaussian_matrix)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        gaussian_key = prefix + "gaussian_matrix"
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if gaussian_key not in state_dict and gaussian_key in missing_keys:
            missing_keys.remove(gaussian_key)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        projected = coords @ self.gaussian_matrix
        projected = 2.0 * np.pi * projected
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class HDMapEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        points_per_polyline: int,
        query_radius_m: float,
        max_polylines: int = 96,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_local_layers: int = 2,
        num_global_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
        include_direction: bool = True,
        include_on_route: bool = True,
        include_has_traffic_lights: bool = True,
        include_stop_line_subtype: bool = True,
        include_speed_limit: bool = True,
        speed_limit_normalizer_mps: float = 30.0,
    ) -> None:
        super().__init__()
        if points_per_polyline <= 1:
            raise ValueError(f"`points_per_polyline` must be > 1, got {points_per_polyline}.")
        if max_polylines <= 0:
            raise ValueError(f"`max_polylines` must be > 0, got {max_polylines}.")

        try:
            from transformers import BertConfig, BertModel
        except ImportError as exc:
            raise ImportError(
                "HDMapEncoder requires the `transformers` package. Install it in the `drivoR` environment "
                "or add it to your environment setup before enabling `config.hd_map.enabled=true`."
            ) from exc

        self.points_per_polyline = points_per_polyline
        self.include_direction = include_direction
        self.include_on_route = include_on_route
        self.include_has_traffic_lights = include_has_traffic_lights
        self.include_stop_line_subtype = include_stop_line_subtype
        self.include_speed_limit = include_speed_limit

        self.coord_embed = RandomFourierPointEmbedding(embed_dim=embed_dim)
        self.direction_embed = RandomFourierPointEmbedding(embed_dim=embed_dim)
        self.geometry_embed = nn.Embedding(len(HD_MAP_GEOMETRY_TYPE_TO_ID), embed_dim)
        self.element_type_embed = nn.Embedding(len(HD_MAP_ELEMENT_TYPE_TO_ID), embed_dim)
        self.on_route_embed = nn.Embedding(3, embed_dim)
        self.has_traffic_light_embed = nn.Embedding(2, embed_dim)
        self.stop_line_subtype_embed = nn.Embedding(HD_MAP_STOP_LINE_SUBTYPE_UNKNOWN + 1, embed_dim)
        self.has_speed_limit_embed = nn.Embedding(2, embed_dim)
        self.speed_limit_proj = nn.Linear(1, embed_dim)
        self.point_index_embed = nn.Embedding(points_per_polyline + 1, embed_dim)
        self.header_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 1e-6)

        local_config = BertConfig(
            hidden_size=embed_dim,
            num_hidden_layers=num_local_layers,
            num_attention_heads=num_heads,
            intermediate_size=embed_dim * ff_mult,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            hidden_act="gelu",
            max_position_embeddings=points_per_polyline + 1,
            type_vocab_size=2,
            vocab_size=1,
        )
        self.local_encoder = (
            BertModel(local_config, add_pooling_layer=False)
            if num_local_layers > 0
            else None
        )

        global_config = BertConfig(
            hidden_size=embed_dim,
            num_hidden_layers=num_global_layers,
            num_attention_heads=num_heads,
            intermediate_size=embed_dim * ff_mult,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            hidden_act="gelu",
            max_position_embeddings=max_polylines,
            type_vocab_size=2,
            vocab_size=1,
        )
        self.global_encoder = (
            BertModel(global_config, add_pooling_layer=False)
            if num_global_layers > 0
            else None
        )

        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
        )

        coord_scale = torch.tensor([query_radius_m, query_radius_m], dtype=torch.float32)
        self.register_buffer("coord_scale", coord_scale, persistent=False)
        speed_limit_normalizer = torch.tensor(float(speed_limit_normalizer_mps), dtype=torch.float32)
        self.register_buffer("speed_limit_normalizer", speed_limit_normalizer, persistent=False)

    def _encode_valid_polylines(
        self,
        polyline_tokens: torch.Tensor,
        polyline_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.local_encoder is None:
            return polyline_tokens

        batch_size, num_polylines, _, embed_dim = polyline_tokens.shape
        flat_tokens = polyline_tokens.reshape(batch_size * num_polylines, self.points_per_polyline + 1, embed_dim)
        flat_valid_mask = polyline_valid_mask.reshape(batch_size * num_polylines)
        encoded_tokens = flat_tokens.new_zeros(flat_tokens.shape)

        if flat_valid_mask.any():
            local_attention_mask = torch.ones(
                (int(flat_valid_mask.sum().item()), self.points_per_polyline + 1),
                dtype=torch.long,
                device=flat_tokens.device,
            )
            encoded_tokens[flat_valid_mask] = self.local_encoder(
                inputs_embeds=flat_tokens[flat_valid_mask],
                attention_mask=local_attention_mask,
                return_dict=True,
            ).last_hidden_state

        return encoded_tokens.reshape(batch_size, num_polylines, self.points_per_polyline + 1, embed_dim)

    def _encode_global(
        self,
        header_tokens: torch.Tensor,
        polyline_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.global_encoder is None:
            return header_tokens

        encoded_headers = header_tokens.new_zeros(header_tokens.shape)
        non_empty_samples = polyline_valid_mask.any(dim=1)

        if non_empty_samples.any():
            encoded_headers[non_empty_samples] = self.global_encoder(
                inputs_embeds=header_tokens[non_empty_samples],
                attention_mask=polyline_valid_mask[non_empty_samples].long(),
                return_dict=True,
            ).last_hidden_state

        return encoded_headers

    def _attribute_bias(
        self,
        geometry_type_ids: torch.Tensor,
        element_type_ids: torch.Tensor,
        on_route_ids: torch.Tensor,
        has_traffic_light_ids: torch.Tensor,
        stop_line_subtype_ids: torch.Tensor,
        speed_limit_mps: torch.Tensor,
    ) -> torch.Tensor:
        bias = self.geometry_embed(geometry_type_ids) + self.element_type_embed(element_type_ids)
        if self.include_on_route:
            bias = bias + self.on_route_embed(on_route_ids)
        if self.include_has_traffic_lights:
            bias = bias + self.has_traffic_light_embed(has_traffic_light_ids)
        if self.include_stop_line_subtype:
            bias = bias + self.stop_line_subtype_embed(stop_line_subtype_ids)
        if self.include_speed_limit:
            has_speed_limit_ids = (speed_limit_mps >= 0.0).long()
            clamped_speed_limit = speed_limit_mps.clamp(min=0.0).unsqueeze(-1)
            norm_denominator = self.speed_limit_normalizer.to(
                device=clamped_speed_limit.device,
                dtype=clamped_speed_limit.dtype,
            )
            normalized_speed_limit = clamped_speed_limit / norm_denominator.clamp(min=1e-6)
            bias = bias + self.has_speed_limit_embed(has_speed_limit_ids) + self.speed_limit_proj(normalized_speed_limit)
        return bias

    def forward(
        self,
        coords: torch.Tensor,
        direction_vectors: torch.Tensor,
        valid_mask: torch.Tensor,
        geometry_type_ids: torch.Tensor,
        element_type_ids: torch.Tensor,
        on_route_ids: torch.Tensor,
        has_traffic_light_ids: torch.Tensor,
        stop_line_subtype_ids: torch.Tensor,
        speed_limit_mps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return:
            map_tokens: [B, num_polylines, output_dim]
            padding_mask: [B, num_polylines], True for padded / invalid polylines
        """
        if coords.ndim != 4:
            raise ValueError(f"`coords` must have shape [B, P, N, 2], got {tuple(coords.shape)}.")
        if direction_vectors.shape != coords.shape:
            raise ValueError(
                "`direction_vectors` must have the same shape as `coords`, "
                f"got {tuple(direction_vectors.shape)} vs {tuple(coords.shape)}."
            )

        coord_scale = self.coord_scale.to(device=coords.device, dtype=coords.dtype)
        norm_coords = coords / coord_scale
        direction_vectors = F.normalize(direction_vectors.to(device=coords.device, dtype=coords.dtype), dim=-1, eps=1e-6)

        point_tokens = self.coord_embed(norm_coords)
        if self.include_direction:
            point_tokens = point_tokens + self.direction_embed(direction_vectors)
        attr_bias = self._attribute_bias(
            geometry_type_ids=geometry_type_ids,
            element_type_ids=element_type_ids,
            on_route_ids=on_route_ids,
            has_traffic_light_ids=has_traffic_light_ids,
            stop_line_subtype_ids=stop_line_subtype_ids,
            speed_limit_mps=speed_limit_mps,
        )

        point_index = torch.arange(self.points_per_polyline + 1, device=coords.device)
        point_index_embed = self.point_index_embed(point_index)

        point_tokens = point_tokens + attr_bias[:, :, None, :] + point_index_embed[1:].view(1, 1, -1, point_tokens.shape[-1])

        header_coords = norm_coords.mean(dim=2)
        header_direction = F.normalize(direction_vectors.mean(dim=2), dim=-1, eps=1e-6)
        header_tokens = (
            self.header_token
            + self.coord_embed(header_coords).unsqueeze(2)
            + attr_bias.unsqueeze(2)
            + point_index_embed[:1].view(1, 1, 1, -1)
        )
        if self.include_direction:
            header_tokens = header_tokens + self.direction_embed(header_direction).unsqueeze(2)

        polyline_tokens = torch.cat([header_tokens, point_tokens], dim=2)
        encoded_polylines = self._encode_valid_polylines(polyline_tokens, valid_mask)
        header_tokens = encoded_polylines[:, :, 0]
        header_tokens = self._encode_global(header_tokens, valid_mask)
        map_tokens = self.output_proj(header_tokens)
        map_tokens = map_tokens.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return map_tokens, ~valid_mask
