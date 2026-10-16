#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""Gaussian parameters and deformation, without training or file I/O concerns."""

from dataclasses import dataclass, is_dataclass

import numpy as np
import torch
from torch import Tensor, nn

from fourdgs.config import ModelConfig
from fourdgs.data.types import PointCloud
from fourdgs.geometry.gaussians import inverse_sigmoid
from fourdgs.geometry.sh import rgb_to_sh
from fourdgs.models.deformation import DeformationNetwork


@dataclass
class GaussianAttributes:
    """Unactivated parameters: XYZ, log scales, wxyz rotations, opacity logits, SH."""

    positions: Tensor
    log_scales: Tensor
    rotations: Tensor
    opacity_logits: Tensor
    sh: Tensor


POINT_PARAMETERS = ("positions", "sh_dc", "sh_rest", "log_scales", "rotations", "opacity_logits")


class GaussianModel(nn.Module):
    """Canonical Gaussians and a time-conditioned deformation network."""

    positions: nn.Parameter
    sh_dc: nn.Parameter
    sh_rest: nn.Parameter
    log_scales: nn.Parameter
    rotations: nn.Parameter
    opacity_logits: nn.Parameter

    def __init__(self, config: ModelConfig):
        super().__init__()
        if is_dataclass(config):
            from omegaconf import OmegaConf

            config = OmegaConf.structured(config)
        self.config = config
        self.active_sh_degree = 0
        self.spatial_lr_scale = 1.0
        self.deformation = DeformationNetwork(config)
        coefficients = (config.sh_degree + 1) ** 2
        for name, shape in zip(
            POINT_PARAMETERS,
            [(0, 3), (0, 1, 3), (0, coefficients - 1, 3), (0, 3), (0, 4), (0, 1)],
            strict=True,
        ):
            setattr(self, name, nn.Parameter(torch.empty(shape)))

    @property
    def device(self) -> torch.device:
        return self.positions.device

    @property
    def count(self) -> int:
        return self.positions.shape[0]

    def point_parameters(self) -> dict[str, nn.Parameter]:
        return {name: getattr(self, name) for name in POINT_PARAMETERS}

    def replace_points(self, parameters: dict[str, Tensor]) -> None:
        """Set parameters before creating an optimizer (initialization or loading)."""
        counts = {len(parameters[name]) for name in POINT_PARAMETERS}
        if len(counts) != 1:
            raise ValueError("All Gaussian attributes must have equal point counts")
        for name in POINT_PARAMETERS:
            setattr(self, name, nn.Parameter(parameters[name].contiguous()))

    def initialize(self, cloud: PointCloud, radius: float, device: str | torch.device) -> None:
        """Initialize scales from the mean squared distance to three neighbors."""
        if len(cloud.points) < 4:
            raise ValueError("Gaussian initialization requires at least four points")
        from simple_knn._C import distCUDA2

        self.to(device)
        positions = torch.tensor(np.asarray(cloud.points), dtype=torch.float32, device=device)
        colors = rgb_to_sh(
            torch.tensor(np.asarray(cloud.colors), dtype=torch.float32, device=device)
        )
        sh = torch.zeros((len(positions), 3, (self.config.sh_degree + 1) ** 2), device=device)
        sh[:, :, 0] = colors
        distances = distCUDA2(positions.contiguous()).clamp_min(1e-7)
        rotations = torch.zeros((len(positions), 4), device=device)
        rotations[:, 0] = 1
        self.replace_points(
            dict(
                positions=positions,
                sh_dc=sh[:, :, :1].transpose(1, 2),
                sh_rest=sh[:, :, 1:].transpose(1, 2),
                log_scales=distances.sqrt().log()[:, None].repeat(1, 3),
                rotations=rotations,
                opacity_logits=inverse_sigmoid(torch.full((len(positions), 1), 0.1, device=device)),
            )
        )
        self.spatial_lr_scale = radius
        self.deformation.deformation_net.set_aabb(cloud.points.max(0), cloud.points.min(0))

    def attributes(self, time: float | Tensor, stage: str = "fine") -> GaussianAttributes:
        """Evaluate raw Gaussian attributes without applying renderer activations."""
        sh = torch.cat((self.sh_dc, self.sh_rest), dim=1)
        values = self.positions, self.log_scales, self.rotations, self.opacity_logits, sh
        if stage == "fine":
            times = torch.as_tensor(time, dtype=self.positions.dtype, device=self.device).reshape(
                -1, 1
            )
            times = times.expand(self.count, 1)
            values = self.deformation(*values, times)
        elif stage != "coarse":
            raise ValueError(f"Unknown training stage: {stage}")
        return GaussianAttributes(*values)

    def increase_sh_degree(self) -> None:
        self.active_sh_degree = min(self.active_sh_degree + 1, self.config.sh_degree)
