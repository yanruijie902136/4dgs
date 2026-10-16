"""Optional dense occupancy field for the original deformation ablation."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DenseGrid(nn.Module):
    """Trilinearly interpolate an occupancy mask within registered world bounds."""

    xyz_min: Tensor
    xyz_max: Tensor

    def __init__(self, channels: int, world_size: Sequence[int]):
        super().__init__()
        self.channels = channels
        self.world_size = tuple(world_size)
        self.grid = nn.Parameter(torch.ones([1, channels, *world_size]))
        self.register_buffer("xyz_min", torch.full((3,), -1.0))
        self.register_buffer("xyz_max", torch.full((3,), 1.0))

    def forward(self, positions: Tensor) -> Tensor:
        """Return one mask feature vector for each world-space XYZ position."""
        shape = positions.shape[:-1]
        positions = positions.reshape(1, 1, 1, -1, 3)
        normalized = ((positions - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
            (-1,)
        ) * 2 - 1
        sampled = F.grid_sample(self.grid, normalized, mode="bilinear", align_corners=True)
        return sampled.reshape(self.channels, -1).T.reshape(*shape, self.channels)

    def set_aabb(self, maximum, minimum) -> None:
        """Replace world bounds without changing their device or buffer ownership."""
        self.xyz_min.copy_(torch.as_tensor(minimum, dtype=self.grid.dtype, device=self.grid.device))
        self.xyz_max.copy_(torch.as_tensor(maximum, dtype=self.grid.dtype, device=self.grid.device))

    def extra_repr(self) -> str:
        return f"channels={self.channels}, world_size={self.world_size}"
