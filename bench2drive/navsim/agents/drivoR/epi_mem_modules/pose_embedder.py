# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

import torch
from torch import nn

class PoseEmbedder(nn.Module):
    def __init__(self, norm_range, dim=256, num_freqs=6, pose_dim=3):
        """
        Args:
            norm_range: Per-axis normalization range in meters.
            dim: Hidden and output dimension of the MLP.
            num_freqs: Number of frequency bands for NeRF positional encoding.
            pose_dim: Pose embedding dimensionality mode. 2 => planar [x, y], 3 => [x, y, z].
        """
        
        super().__init__()
        self.num_freqs = num_freqs
        self.dim = dim
        self.pose_dim = int(pose_dim)
        if self.pose_dim not in (2, 3):
            raise ValueError(f"pose_dim must be 2 or 3, got {self.pose_dim}")

        norm_range = torch.as_tensor(norm_range, dtype=torch.float32).flatten()
        if norm_range.numel() < self.pose_dim:
            raise ValueError(
                f"norm_range must have at least {self.pose_dim} elements for pose_dim={self.pose_dim}, "
                f"got {norm_range.numel()}"
            )
        norm_range = norm_range[: self.pose_dim]
        self.norm_range = nn.Parameter(norm_range, requires_grad=False)

        # Input elements: 2D => [2x3]=6 from [R2x2 | t_xy], 3D => [3x4]=12 from [R3x3 | t_xyz]
        base_feat_dim = 6 if self.pose_dim == 2 else 12
        self.input_feat_dim = base_feat_dim * 2 * num_freqs 

        self.mlp = nn.Sequential(
            nn.Linear(self.input_feat_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, rel_pose_matrix):
        """
        Args:
            rel_pose_matrix: Tensor (B, 3, 4) or (B, 4, 4)
                             Relative transform Memory -> Current (i.e. memory expressed in current frame).
        """
        if self.pose_dim == 2:
            # Keep only planar components [R2x2 | t_xy] => (..., 2, 3).
            affine = rel_pose_matrix[..., :2, :]
            affine = affine[..., [0, 1, 3]]
        else:
            # Keep full affine [R3x3 | t_xyz] => (..., 3, 4).
            affine = rel_pose_matrix[..., :3, :]
        
        # Rotation terms are already approximately [-1, 1]. Normalize translation in meters.
        affine_norm = affine.clone()
        affine_norm[..., -1] /= self.norm_range

        # NeRF Encoding
        encoded_pose = nerf_positional_encoding(
            affine_norm.flatten(-2), 
            num_encoding_functions=self.num_freqs, 
            include_input=False
        )
        
        return self.mlp(encoded_pose)




def nerf_positional_encoding(
    tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)
