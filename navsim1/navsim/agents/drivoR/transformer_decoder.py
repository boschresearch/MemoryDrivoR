# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import numpy as np
from torch.jit import Final
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from .layers.utils.mlp import MLP
from .timm_layers import (
    Mlp,
    DropPath,
    LayerScale,
)


class Attention(torch.nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.proj_drop = torch.nn.Dropout(proj_drop)

        self.attn = torch.nn.MultiheadAttention(dim, num_heads, 
                                                dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=True, device=None, dtype=None)


    def forward(
        self,
        q: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if kv is None:
            x = self.attn(query=q, key=q, value=q, need_weights=False)[0]
        else:
            empty_samples = False
            if key_padding_mask is not None:
                fully_masked_rows = key_padding_mask.all(dim=1)
                empty_samples = fully_masked_rows.any()

            if kv.shape[1] == 0:
                x = q.new_zeros(q.shape)
            elif empty_samples:
                x = q.new_zeros(q.shape)
                valid_rows = ~fully_masked_rows
                if valid_rows.any():
                    valid_out = self.attn(
                        query=q[valid_rows],
                        key=kv[valid_rows],
                        value=kv[valid_rows],
                        need_weights=False,
                        key_padding_mask=key_padding_mask[valid_rows],
                    )[0]
                    x[valid_rows] = valid_out.to(dtype=x.dtype)
            else:
                x = self.attn(
                    query=q,
                    key=kv,
                    value=kv,
                    need_weights=False,
                    key_padding_mask=key_padding_mask,
                )[0]
        x = self.proj_drop(x)
        return x

class Block(torch.nn.Module):


    def __init__(self,
            dim: int,
            num_heads: int,
            num_extra_cross_attns: int = 0,
            mlp_ratio: float = 4.,
            scale_mlp_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            drop_path: float = 0.,
            init_values: float = 0.0,
            act_layer: Type[torch.nn.Module] = torch.nn.GELU,
            norm_layer: Type[torch.nn.Module] = torch.nn.LayerNorm,
            mlp_layer: Type[torch.nn.Module] = Mlp,):
        super().__init__()

        # self attention layer
        self.self_attn_norm = norm_layer(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            proj_drop=proj_drop,
        )
        self.self_attn_ls = LayerScale(dim, init_values=init_values) if (init_values > 0) else torch.nn.Identity()
        self.self_attn_drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()

        # cross attention network
        self.cross_attn_norm_kv = norm_layer(dim)
        self.cross_attn_norm_q = norm_layer(dim)
        self.cross_attn = Attention(
            dim,
            num_heads=num_heads,
            proj_drop=proj_drop,
        )
        self.cross_attn_ls = LayerScale(dim, init_values=init_values) if (init_values > 0) else torch.nn.Identity()
        self.cross_attn_drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        self.extra_cross_attn_norm_kv = torch.nn.ModuleList(
            [norm_layer(dim) for _ in range(num_extra_cross_attns)]
        )
        self.extra_cross_attn_norm_q = torch.nn.ModuleList(
            [norm_layer(dim) for _ in range(num_extra_cross_attns)]
        )
        self.extra_cross_attn = torch.nn.ModuleList(
            [
                Attention(
                    dim,
                    num_heads=num_heads,
                    proj_drop=proj_drop,
                )
                for _ in range(num_extra_cross_attns)
            ]
        )
        self.extra_cross_attn_ls = torch.nn.ModuleList(
            [
                LayerScale(dim, init_values=init_values) if (init_values > 0) else torch.nn.Identity()
                for _ in range(num_extra_cross_attns)
            ]
        )
        self.extra_cross_attn_drop_path = torch.nn.ModuleList(
            [DropPath(drop_path) if drop_path > 0. else torch.nn.Identity() for _ in range(num_extra_cross_attns)]
        )

        # create the FFN network
        self.mlp_norm = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer if scale_mlp_norm else None,
            bias=proj_bias,
            drop=proj_drop,
            )
        self.mlp_ls = LayerScale(dim, init_values=init_values) if init_values else torch.nn.Identity()
        self.mlp_drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()



    def _apply_cross_attention(
        self,
        x: torch.Tensor,
        x_cross: Optional[torch.Tensor],
        x_cross_padding_mask: Optional[torch.Tensor],
        q_norm: torch.nn.Module,
        kv_norm: torch.nn.Module,
        attn: torch.nn.Module,
        layer_scale: torch.nn.Module,
        drop_path: torch.nn.Module,
    ) -> torch.Tensor:
        if x_cross is None:
            return x
        return x + drop_path(
            layer_scale(
                attn(
                    q_norm(x),
                    kv_norm(x_cross),
                    key_padding_mask=x_cross_padding_mask,
                )
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        x_cross: torch.Tensor,
        x_cross_padding_mask: Optional[torch.Tensor] = None,
        extra_cross: Optional[List[Optional[torch.Tensor]]] = None,
        extra_cross_padding_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if extra_cross is None:
            extra_cross = []
        if extra_cross_padding_masks is None:
            extra_cross_padding_masks = [None] * len(extra_cross)
        if len(extra_cross) != len(self.extra_cross_attn):
            raise ValueError(
                f"Expected {len(self.extra_cross_attn)} extra cross-attention inputs, got {len(extra_cross)}."
            )
        if len(extra_cross_padding_masks) != len(extra_cross):
            raise ValueError(
                "extra_cross_padding_masks must match extra_cross length: "
                f"{len(extra_cross_padding_masks)} vs {len(extra_cross)}."
            )

        x = x + self.self_attn_drop_path(
                        self.self_attn_ls(
                            self.self_attn(
                                self.self_attn_norm(x))))

        x = self._apply_cross_attention(
            x,
            x_cross=x_cross,
            x_cross_padding_mask=x_cross_padding_mask,
            q_norm=self.cross_attn_norm_q,
            kv_norm=self.cross_attn_norm_kv,
            attn=self.cross_attn,
            layer_scale=self.cross_attn_ls,
            drop_path=self.cross_attn_drop_path,
        )
        for extra_kv, extra_mask, q_norm, kv_norm, attn, layer_scale, drop_path in zip(
            extra_cross,
            extra_cross_padding_masks,
            self.extra_cross_attn_norm_q,
            self.extra_cross_attn_norm_kv,
            self.extra_cross_attn,
            self.extra_cross_attn_ls,
            self.extra_cross_attn_drop_path,
        ):
            x = self._apply_cross_attention(
                x,
                x_cross=extra_kv,
                x_cross_padding_mask=extra_mask,
                q_norm=q_norm,
                kv_norm=kv_norm,
                attn=attn,
                layer_scale=layer_scale,
                drop_path=drop_path,
            )

        x = x + self.mlp_drop_path(
                        self.mlp_ls(
                            self.mlp(
                                self.mlp_norm(x))))
        
        return x

class TransformerDecoder(torch.nn.Module):

    def __init__(self, proj_drop, drop_path, config, num_extra_cross_attns: int = 0):
        super().__init__()

        num_layers = config.ref_num
        d_model = config.tf_d_model

        _layers = []
        for i in range(num_layers):
            _layers.append(
                Block(
                    dim=d_model,
                    num_heads=config.refiner_num_heads if hasattr(config, "refiner_num_heads") else 1,
                    num_extra_cross_attns=num_extra_cross_attns,
                    init_values= config.refiner_ls_values if hasattr(config, "refiner_ls_values") else 0.0,
                    proj_drop=proj_drop,
                    drop_path=drop_path
                )
            )
        self.layers = torch.nn.ModuleList(_layers)
        self.return_intermediate = True


    def forward(
        self,
        x,
        x_cross,
        x_cross_padding_mask: Optional[torch.Tensor] = None,
        extra_cross: Optional[List[Optional[torch.Tensor]]] = None,
        extra_cross_padding_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ):
        
        intermediate = []
        for _, layer in enumerate(self.layers):
            x = layer(
                x,
                x_cross,
                x_cross_padding_mask=x_cross_padding_mask,
                extra_cross=extra_cross,
                extra_cross_padding_masks=extra_cross_padding_masks,
            )
            if self.return_intermediate:
                intermediate.append(x)

        if self.return_intermediate:
            return torch.stack(intermediate)
        else:
            return x


class TransformerDecoderScorer(torch.nn.Module):

    def __init__(self, num_layers, d_model, proj_drop, drop_path, config, num_extra_cross_attns: int = 0):
        super().__init__()

        _layers = []
        for i in range(num_layers):
            _layers.append(
                Block(
                    dim=d_model,
                    num_heads=config.refiner_num_heads if hasattr(config, "refiner_num_heads") else 1,
                    num_extra_cross_attns=num_extra_cross_attns,
                    init_values= config.refiner_ls_values if hasattr(config, "refiner_ls_values") else 0.0,
                    proj_drop=proj_drop,
                    drop_path=drop_path
                )
            )
        self.layers = torch.nn.ModuleList(_layers)
        self.return_intermediate = False


    def forward(
        self,
        x,
        x_cross,
        x_cross_padding_mask: Optional[torch.Tensor] = None,
        extra_cross: Optional[List[Optional[torch.Tensor]]] = None,
        extra_cross_padding_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ):
        
        intermediate = []
        for _, layer in enumerate(self.layers):
            x = layer(
                x,
                x_cross,
                x_cross_padding_mask=x_cross_padding_mask,
                extra_cross=extra_cross,
                extra_cross_padding_masks=extra_cross_padding_masks,
            )
            if self.return_intermediate:
                intermediate.append(x)

        if self.return_intermediate:
            return torch.stack(intermediate)
        else:
            return x
