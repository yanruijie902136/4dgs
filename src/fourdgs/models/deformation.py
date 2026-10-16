"""HexPlane-conditioned deformation of raw Gaussian attributes.

State-dict names match original 4DGS snapshots. The dormant time network and
encoding buffers are retained solely for original initialization/weight import;
the dynamic path used only the unencoded XYZ and time coordinates upstream.
"""

import torch
from torch import Tensor, nn

from fourdgs.geometry.gaussians import quaternion_multiply
from fourdgs.models.grid import DenseGrid
from fourdgs.models.hexplane import HexPlaneField


def initialize_weights(module: nn.Module) -> None:
    """Retain the original two Xavier draws for biased linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            nn.init.xavier_uniform_(module.weight, gain=1)


class DeformationMLP(nn.Module):
    """Map plane features to position, scale, rotation, opacity, and SH offsets."""

    pos_deform: nn.Sequential
    scales_deform: nn.Sequential
    rotations_deform: nn.Sequential
    opacity_deform: nn.Sequential
    shs_deform: nn.Sequential

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grid = HexPlaneField(config.bounds, config.kplanes_config, config.multires)
        width = config.net_width
        if config.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64, 64, 64])
        if config.static_mlp:
            self.static_mlp = nn.Sequential(
                nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1)
            )
        input_features = 4 if config.no_grid else self.grid.feat_dim * (1 + 2 * config.grid_pe)
        layers: list[nn.Module] = [nn.Linear(input_features, width)]
        for _ in range(config.defor_depth - 1):
            layers.extend((nn.ReLU(), nn.Linear(width, width)))
        self.feature_out = nn.Sequential(*layers)
        for name, output_features in (
            ("pos_deform", 3),
            ("scales_deform", 3),
            ("rotations_deform", 4),
            ("opacity_deform", 1),
            ("shs_deform", 48),
        ):
            setattr(
                self,
                name,
                nn.Sequential(
                    nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, output_features)
                ),
            )

    def set_aabb(self, maximum, minimum) -> None:
        """Install world bounds using the original maximum/minimum ordering."""
        self.grid.set_aabb(maximum, minimum)
        if self.config.empty_voxel:
            self.empty_voxel.set_aabb(maximum, minimum)

    def features(self, positions: Tensor, times: Tensor) -> Tensor:
        if self.config.no_grid:
            values = torch.cat((positions, times), dim=-1)
        else:
            values = self.grid(positions, times)
            if self.config.grid_pe:
                frequencies = 2.0 ** torch.arange(self.config.grid_pe, device=values.device)
                encoded = (values.unsqueeze(-1) * frequencies).flatten(-2)
                values = torch.cat((values, encoded.sin(), encoded.cos()), dim=-1)
        return self.feature_out(values)

    def forward(
        self,
        positions: Tensor,
        scales: Tensor,
        rotations: Tensor,
        opacity: Tensor,
        sh: Tensor,
        times: Tensor,
    ) -> tuple[Tensor, ...]:
        hidden = self.features(positions, times)
        config = self.config
        if config.static_mlp:
            mask = self.static_mlp(hidden)
        elif config.empty_voxel:
            mask = self.empty_voxel(positions)
        else:
            mask = torch.ones_like(opacity)
        if not config.no_dx:
            positions = positions * mask + self.pos_deform(hidden)
        if not config.no_ds:
            scales = scales * mask + self.scales_deform(hidden)
        if not config.no_dr:
            delta = self.rotations_deform(hidden)
            rotations = (
                quaternion_multiply(rotations, delta)
                if config.apply_rotation
                else rotations + delta
            )
        if not config.no_do:
            opacity = opacity * mask + self.opacity_deform(hidden)
        if not config.no_dshs:
            delta = self.shs_deform(hidden).reshape(len(sh), 16, 3)[:, : sh.shape[1]]
            sh = sh * mask.unsqueeze(-1) + delta
        return positions, scales, rotations, opacity, sh


class DeformationNetwork(nn.Module):
    """Deformation field with parameter groups compatible with original weights."""

    def __init__(self, config):
        super().__init__()
        self.timenet = nn.Sequential(
            nn.Linear(2 * config.timebase_pe + 1, config.timenet_width),
            nn.ReLU(),
            nn.Linear(config.timenet_width, config.timenet_output),
        )
        self.deformation_net = DeformationMLP(config)
        for name, count in (
            ("time_poc", config.timebase_pe),
            ("pos_poc", config.posebase_pe),
            ("rotation_scaling_poc", config.scale_rotation_pe),
            ("opacity_poc", config.opacity_pe),
        ):
            self.register_buffer(
                name, torch.tensor([2**i for i in range(count)], dtype=torch.float32)
            )
        self.apply(initialize_weights)

    def forward(
        self,
        positions: Tensor,
        scales: Tensor,
        rotations: Tensor,
        opacity: Tensor,
        sh: Tensor,
        times: Tensor,
    ) -> tuple[Tensor, ...]:
        return self.deformation_net(positions, scales, rotations, opacity, sh, times)

    def get_mlp_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.deformation_net.named_parameters()
            if "grid" not in name
        ] + list(self.timenet.parameters())

    def get_grid_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.deformation_net.named_parameters()
            if "grid" in name
        ]
