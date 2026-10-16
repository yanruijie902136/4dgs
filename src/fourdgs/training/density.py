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

"""Original clone/split/prune policy, separated from model and optimizer ownership."""

import torch
from torch import Tensor

from fourdgs.config import DensityConfig
from fourdgs.geometry.gaussians import build_rotation, inverse_sigmoid
from fourdgs.models.gaussian import GaussianModel
from fourdgs.training.optimizer import OptimizerController


class DensityController:
    """Accumulate screen-space gradients and adjust Gaussian topology."""

    def __init__(self, model: GaussianModel, optimizer: OptimizerController, config: DensityConfig):
        self.model, self.optimizer, self.config = model, optimizer, config
        self.reset_statistics()

    def reset_statistics(self) -> None:
        count, device = self.model.count, self.model.device
        self.gradient_sum = torch.zeros((count, 1), device=device)
        self.observations = torch.zeros((count, 1), device=device)
        self.max_radii = torch.zeros(count, device=device)

    def state_dict(self) -> dict[str, Tensor]:
        return {name: getattr(self, name) for name in ("gradient_sum", "observations", "max_radii")}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for name in self.state_dict():
            value = state[name].to(self.model.device)
            if len(value) != self.model.count:
                raise ValueError(f"Density state {name} has incorrect point count")
            setattr(self, name, value)

    @torch.no_grad()
    def observe(self, gradients: Tensor, visible: Tensor, radii: Tensor) -> None:
        self.max_radii[visible] = torch.maximum(self.max_radii[visible], radii[visible])
        self.gradient_sum[visible] += gradients[visible, :2].norm(dim=-1, keepdim=True)
        self.observations[visible] += 1

    @torch.no_grad()
    def append(self, values: dict[str, Tensor]) -> None:
        count = len(values["positions"])
        joined = {
            name: torch.cat((parameter, values[name]))
            for name, parameter in self.model.point_parameters().items()
        }
        self.optimizer.replace(joined, append=count)
        # Upstream resets accumulated statistics after every clone/split pass.
        self.reset_statistics()

    @torch.no_grad()
    def prune(self, remove: Tensor) -> None:
        keep = ~remove
        if not keep.any():
            # Keep the most opaque Gaussian rather than producing an invalid scene.
            keep[self.model.opacity_logits.flatten().argmax()] = True
        self.optimizer.replace(
            {name: parameter[keep] for name, parameter in self.model.point_parameters().items()},
            keep=keep,
        )
        for name, value in self.state_dict().items():
            setattr(self, name, value[keep])

    @torch.no_grad()
    def densify(self, threshold: float, extent: float) -> None:
        model = self.model
        gradients = torch.nan_to_num(self.gradient_sum / self.observations, nan=0.0)
        small = model.log_scales.exp().max(dim=1).values <= self.config.percent_dense * extent
        selected = (gradients.norm(dim=-1) >= threshold) & small
        self.append({name: value[selected] for name, value in model.point_parameters().items()})
        padded = torch.zeros(model.count, device=model.device)
        padded[: len(gradients)] = gradients.flatten()
        selected = (padded >= threshold) & (
            model.log_scales.exp().max(dim=1).values > self.config.percent_dense * extent
        )
        if not selected.any():
            return
        count = int(selected.sum())
        values = {
            name: value[selected].repeat((2,) + (1,) * (value.ndim - 1))
            for name, value in model.point_parameters().items()
        }
        std = model.log_scales[selected].exp().repeat(2, 1)
        samples = torch.normal(torch.zeros_like(std), std)
        rotations = build_rotation(model.rotations[selected]).repeat(2, 1, 1)
        values["positions"] += (rotations @ samples.unsqueeze(-1)).squeeze(-1)
        values["log_scales"] = (std / 1.6).log()
        self.append(values)
        self.prune(
            torch.cat((selected, torch.zeros(2 * count, dtype=torch.bool, device=model.device)))
        )

    @torch.no_grad()
    def update(self, step: int, stage: str, extent: float) -> None:
        config, model = self.config, self.model
        if step >= config.densify_until_iter:
            return
        if stage == "coarse":
            opacity, gradient = (
                config.opacity_threshold_coarse,
                config.densify_grad_threshold_coarse,
            )
        else:
            fraction = step / config.densify_until_iter
            opacity = (
                config.opacity_threshold_fine_init * (1 - fraction)
                + config.opacity_threshold_fine_after * fraction
            )
            gradient = (
                config.densify_grad_threshold_fine_init * (1 - fraction)
                + config.densify_grad_threshold_after * fraction
            )
        if (
            step > config.densify_from_iter
            and step % config.densification_interval == 0
            and model.count < config.max_points
        ):
            self.densify(gradient, extent)
        if (
            step > config.pruning_from_iter
            and step % config.pruning_interval == 0
            and model.count > config.prune_min_points
        ):
            remove = model.opacity_logits.sigmoid().flatten() < opacity
            if step > config.opacity_reset_interval:
                remove |= (self.max_radii > 20) | (
                    model.log_scales.exp().max(1).values > 0.1 * extent
                )
            self.prune(remove)
        if step % config.opacity_reset_interval == 0:
            logits = inverse_sigmoid(model.opacity_logits.sigmoid().clamp_max(0.01))
            self.optimizer.replace({"opacity_logits": logits}, reset_moments=True)
