"""Multiresolution spatial-temporal plane interpolation with original axis ordering."""

from itertools import combinations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def sample_plane(plane: Tensor, coordinates: Tensor) -> Tensor:
    """Bilinearly interpolate [N, 2] normalized coordinates into [N, channels]."""
    samples = F.grid_sample(
        plane,
        coordinates.reshape(1, 1, -1, 2),
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    )
    return samples.reshape(plane.shape[1], -1).T


def initialize_planes(channels: int, resolution: list[int]) -> nn.ParameterList:
    """Initialize spatial planes uniformly and temporal planes to one.

    Plane order is XY, XZ, XT, YZ, YT, ZT, matching original weight snapshots.
    PyTorch stores the two spatial dimensions in reverse coordinate order.
    """
    planes = nn.ParameterList()
    for axes in combinations(range(4), 2):
        plane = nn.Parameter(torch.empty(1, channels, *(resolution[axis] for axis in axes[::-1])))
        if 3 in axes:
            nn.init.ones_(plane)
        else:
            nn.init.uniform_(plane, a=0.1, b=0.5)
        planes.append(plane)
    return planes


class HexPlaneField(nn.Module):
    """Multiply six plane samples at each scale, then concatenate scale features."""

    def __init__(self, bounds: float, plane_config, multires: list[int]) -> None:
        super().__init__()
        if plane_config.grid_dimensions != 2 or plane_config.input_coordinate_dim != 4:
            raise ValueError("HexPlane requires six 2D planes over XYZ and time")
        # Preserve the upstream reversed maximum/minimum ordering for weight compatibility.
        self.aabb = nn.Parameter(torch.tensor([[bounds] * 3, [-bounds] * 3]), requires_grad=False)
        self.grids = nn.ModuleList()
        channels = plane_config.output_coordinate_dim
        for multiplier in multires:
            resolution = [value * multiplier for value in plane_config.resolution[:3]]
            resolution += list(plane_config.resolution[3:])
            self.grids.append(initialize_planes(channels, resolution))
        self.feat_dim = channels * len(multires)

    def set_aabb(self, maximum, minimum) -> None:
        """Install scene bounds on the existing parameter's device."""
        bounds = torch.stack(
            [
                torch.as_tensor(value, dtype=self.aabb.dtype, device=self.aabb.device)
                for value in (maximum, minimum)
            ]
        )
        self.aabb = nn.Parameter(bounds, requires_grad=False)

    def forward(self, positions: Tensor, timestamps: Tensor) -> Tensor:
        """Interpolate features for [N, 3] world positions and [N, 1] times."""
        normalized = (positions - self.aabb[0]) * (2.0 / (self.aabb[1] - self.aabb[0])) - 1.0
        coordinates = torch.cat((normalized, timestamps), dim=-1)
        features = []
        for planes in self.grids:
            product: Tensor | float = 1.0
            for plane, axes in zip(planes, combinations(range(4), 2), strict=True):
                product = product * sample_plane(plane, coordinates[..., axes])
            assert isinstance(product, Tensor)
            features.append(product)
        return torch.cat(features, dim=-1)
