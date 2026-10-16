"""4DGS reconstruction loss and spatial/temporal plane regularization."""

import torch
from torch import Tensor

from fourdgs.config import LossConfig
from fourdgs.evaluation.image import l1_loss, ssim
from fourdgs.models.gaussian import GaussianModel


def plane_smoothness(plane: Tensor) -> Tensor:
    """Mean squared second difference along the original plane's height axis."""
    if plane.shape[-2] < 3:
        return plane.sum() * 0
    return plane.diff(dim=-2).diff(dim=-2).square().mean()


def regularization(model: GaussianModel, config: LossConfig) -> Tensor:
    spatial = temporal = l1 = model.positions.new_zeros(())
    for planes in model.deformation.deformation_net.grid.grids:
        if len(planes) != 6:
            continue
        spatial = spatial + sum(plane_smoothness(planes[i]) for i in (0, 1, 3))
        temporal = temporal + sum(plane_smoothness(planes[i]) for i in (2, 4, 5))
        l1 = l1 + sum((1 - planes[i]).abs().mean() for i in (2, 4, 5))
    return (
        config.plane_tv_weight * spatial
        + config.time_smoothness_weight * temporal
        + config.l1_time_planes * l1
    )


def training_loss(
    prediction: Tensor, target: Tensor, model: GaussianModel, config: LossConfig, stage: str
) -> tuple[Tensor, dict[str, Tensor]]:
    """Preserve upstream L1 + weighted DSSIM and fine-stage regularization."""
    reconstruction = l1_loss(prediction, target)
    total = reconstruction
    if stage == "fine" and config.time_smoothness_weight != 0:
        total = total + regularization(model, config)
    if config.lambda_dssim:
        total = total + config.lambda_dssim * (1 - ssim(prediction, target))
    with torch.no_grad():
        psnr = (-10 * torch.log10((prediction - target).square().flatten(1).mean(1))).mean()
    return total, {"l1": reconstruction.detach(), "loss": total.detach(), "psnr": psnr}
