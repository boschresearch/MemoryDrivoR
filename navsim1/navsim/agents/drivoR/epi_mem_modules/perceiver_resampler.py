# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn.functional as F
from torch import nn


class _PerceiverFeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: int):
        super().__init__()
        hidden_dim = dim * ff_mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _PerceiverAttention(nn.Module):
    """
    Perceiver cross-attention:
      Q from latents, K/V from concat(inputs, latents).
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = int(num_heads)
        self.dim_head = dim // self.num_heads
        self.inner_dim = dim

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.dim_head).transpose(1, 2)

    def forward(self, inputs: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        inputs = self.norm_inputs(inputs)
        latents = self.norm_latents(latents)

        q = self.to_q(latents)
        kv_input = torch.cat([inputs, latents], dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(latents.shape[0], latents.shape[1], self.inner_dim)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim=256,
        num_latents=16,
        num_heads=8,
        num_layers=2,
        ff_mult=4,
        version: str = "original",
    ):
        super().__init__()
        
        self.dim = dim
        self.num_latents = num_latents
        self.version = version

        # Learnable latent queries: (num_latents, dim)
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        if version == "decoder":
            # Transformer-decoder resampler.
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * ff_mult,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.layers = None
        elif version == "original":
            # Perceiver attention resampler:
            # latents = latents + Attn(Q=latents, KV=concat(inputs, latents))
            # latents = latents + FFN(latents)
            self.decoder = None
            self.layers = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            _PerceiverAttention(dim=dim, num_heads=num_heads),
                            _PerceiverFeedForward(dim=dim, ff_mult=ff_mult),
                        ]
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown resampler version '{version}'. Expected 'decoder' or 'original'.")

        self.norm_out = nn.LayerNorm(dim)

    @staticmethod
    def _prepare_inputs(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.flatten(1, 2)
        if x.ndim != 3:
            raise ValueError(f"Expected [B, N, D] or [B, T, S, D] input, got shape {tuple(x.shape)}.")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim] or [B, T, S, dim] input features.

        Returns:
            latents: [B, num_latents, dim]
        """
        b = int(x.shape[0])
        if b == 0:
            # Return empty batch directly to avoid zero-batch MHA edge cases.
            return x.new_empty((0, self.num_latents, self.dim))

        inputs = self._prepare_inputs(x)
        latents = self.latents.unsqueeze(0).expand(b, -1, -1)

        if self.version == "decoder":
            out = self.decoder(tgt=latents, memory=inputs)
        else:
            out = latents
            for attn, ff in self.layers:
                out = out + attn(inputs=inputs, latents=out)
                out = out + ff(out)

        return self.norm_out(out)
