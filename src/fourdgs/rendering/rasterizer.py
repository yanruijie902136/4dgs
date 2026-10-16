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

"""The CUDA boundary: activated Gaussians and camera metadata become pixels."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from fourdgs.config import RenderConfig
from fourdgs.data.types import RenderCamera
from fourdgs.geometry.gaussians import build_scaling_rotation, strip_symmetric
from fourdgs.geometry.sh import evaluate_sh
from fourdgs.models.gaussian import GaussianAttributes, GaussianModel


@dataclass
class RenderResult:
    image: Tensor
    depth: Tensor
    viewspace_points: Tensor
    radii: Tensor
    visible: Tensor


def rasterize(
    camera: RenderCamera,
    attributes: GaussianAttributes,
    sh_degree: int,
    background: Tensor,
    config: RenderConfig,
    scaling_modifier: float = 1.0,
    override_color: Tensor | None = None,
) -> RenderResult:
    """Render [N, ...] attributes; preserve screen-space gradients for density control."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    positions = attributes.positions
    sh_degree = min(sh_degree, config.max_sh_degree)
    if positions.device.type != "cuda":
        raise ValueError("Rasterization requires CUDA tensors")
    screen = torch.zeros_like(positions, requires_grad=torch.is_grad_enabled()) + 0
    if screen.requires_grad:
        screen.retain_grad()
    view, projection, center = camera.projection(positions.device)
    settings = GaussianRasterizationSettings(
        image_height=camera.height,
        image_width=camera.width,
        tanfovx=camera.width / (2 * camera.fx),
        tanfovy=camera.height / (2 * camera.fy),
        bg=background,
        scale_modifier=scaling_modifier,
        viewmatrix=view,
        projmatrix=projection,
        sh_degree=sh_degree,
        campos=center,
        prefiltered=False,
        debug=config.debug,
    )
    scales: Tensor | None = attributes.log_scales.exp()
    rotations: Tensor | None = F.normalize(attributes.rotations, dim=-1)
    covariance = None
    if config.compute_covariance_python:
        assert scales is not None and rotations is not None
        transform = build_scaling_rotation(scales * scaling_modifier, rotations)
        covariance = strip_symmetric(transform @ transform.transpose(1, 2))
        scales = rotations = None
    colors = override_color
    sh: Tensor | None = attributes.sh
    if colors is None and config.convert_sh_python:
        assert sh is not None
        direction = F.normalize(positions - center, dim=-1)
        colors = (evaluate_sh(sh_degree, sh.transpose(1, 2), direction) + 0.5).clamp_min(0)
    if colors is not None:
        sh = None
    image, radii, depth = GaussianRasterizer(settings)(
        means3D=positions,
        means2D=screen,
        shs=sh,
        colors_precomp=colors,
        opacities=attributes.opacity_logits.sigmoid(),
        scales=scales,
        rotations=rotations,
        cov3D_precomp=covariance,
    )
    return RenderResult(image, depth, screen, radii, radii > 0)


def render(
    camera: RenderCamera,
    model: GaussianModel,
    config: RenderConfig,
    background: Tensor,
    stage: str = "fine",
    **kwargs,
) -> RenderResult:
    """Evaluate deformation and rasterize one frame."""
    return rasterize(
        camera,
        model.attributes(camera.time, stage),
        model.active_sh_degree,
        background,
        config,
        **kwargs,
    )
