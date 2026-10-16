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

"""Named optimizer groups and atomic point-parameter topology changes."""

import math

import torch
from torch import Tensor, nn

from fourdgs.config import OptimizationConfig
from fourdgs.models.gaussian import GaussianModel


def exponential_lr(initial: float, final: float, step: int, max_steps: int) -> float:
    """Original log-linear schedule (the upstream delay_steps default was zero)."""
    if step < 0 or initial == final == 0:
        return 0.0
    fraction = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(initial) * (1 - fraction) + math.log(final) * fraction)


class OptimizerController:
    """Own Adam state, schedules, and synchronized replacement of Gaussian tensors."""

    def __init__(self, model: GaussianModel, config: OptimizationConfig):
        self.model = model
        self.config = config
        scale = model.spatial_lr_scale
        groups = [
            {
                "params": [model.positions],
                "lr": config.position_lr_init * scale,
                "name": "positions",
            },
            {
                "params": model.deformation.get_mlp_parameters(),
                "lr": config.deformation_lr_init * scale,
                "name": "deformation",
            },
            {
                "params": model.deformation.get_grid_parameters(),
                "lr": config.grid_lr_init * scale,
                "name": "grid",
            },
        ]
        for name, rate in [
            ("sh_dc", config.feature_lr),
            ("sh_rest", config.feature_lr / 20),
            ("opacity_logits", config.opacity_lr),
            ("log_scales", config.scaling_lr),
            ("rotations", config.rotation_lr),
        ]:
            groups.append({"params": [getattr(model, name)], "lr": rate, "name": name})
        self.optimizer = torch.optim.Adam(groups, lr=0, eps=1e-15)

    def update_learning_rates(self, step: int) -> None:
        config = self.config
        schedules = {
            "positions": (config.position_lr_init, config.position_lr_final),
            "deformation": (config.deformation_lr_init, config.deformation_lr_final),
            "grid": (config.grid_lr_init, config.grid_lr_final),
        }
        for group in self.optimizer.param_groups:
            if group["name"] in schedules:
                initial, final = schedules[group["name"]]
                group["lr"] = (
                    exponential_lr(initial, final, step, config.position_lr_max_steps)
                    * self.model.spatial_lr_scale
                )

    @torch.no_grad()
    def replace(
        self,
        values: dict[str, Tensor],
        *,
        keep: Tensor | None = None,
        append: int = 0,
        reset_moments: bool = False,
    ) -> None:
        """Replace selected groups and transform Adam moments using the same topology.

        Deformation groups are identified by name, never by their tensor count.
        New points start with zero moments; surviving points retain their history.
        """
        unknown = values.keys() - self.model.point_parameters().keys()
        if unknown:
            raise ValueError(f"Not point parameters: {unknown}")
        for group in self.optimizer.param_groups:
            name = group["name"]
            if name not in values:
                continue
            old = group["params"][0]
            parameter = nn.Parameter(values[name].contiguous())
            state = self.optimizer.state.pop(old, None)
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key not in state:
                        continue
                    moment = state[key]
                    if reset_moments:
                        moment = torch.zeros_like(parameter)
                    else:
                        if keep is not None:
                            moment = moment[keep]
                        if append:
                            moment = torch.cat((moment, torch.zeros_like(parameter[-append:])))
                    if moment.shape != parameter.shape:
                        raise ValueError(f"Optimizer state shape mismatch for {name}")
                    state[key] = moment
                self.optimizer.state[parameter] = state
            group["params"] = [parameter]
            setattr(self.model, name, parameter)

    def step(self) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
