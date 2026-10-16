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

"""3DGS-compatible raw Gaussian PLY and colored point-cloud serialization."""

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from fourdgs.data.types import PointCloud
from fourdgs.models.gaussian import GaussianAttributes


def write_gaussians(path: str | Path, attributes: GaussianAttributes) -> None:
    """Write log scales, raw quaternions, opacity logits, and channel-major SH."""
    xyz = attributes.positions.detach().cpu().numpy()
    sh = attributes.sh.detach().transpose(1, 2).cpu().numpy()
    dc, rest = sh[:, :, :1].reshape(len(xyz), -1), sh[:, :, 1:].reshape(len(xyz), -1)
    values = [
        xyz,
        np.zeros_like(xyz),
        dc,
        rest,
        attributes.opacity_logits.detach().cpu().numpy(),
        attributes.log_scales.detach().cpu().numpy(),
        attributes.rotations.detach().cpu().numpy(),
    ]
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(dc.shape[1])]
        + [f"f_rest_{i}" for i in range(rest.shape[1])]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    array = np.empty(len(xyz), dtype=[(name, "f4") for name in names])
    combined = np.concatenate(values, axis=1)
    for index, name in enumerate(names):
        array[name] = combined[:, index]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(array, "vertex")]).write(str(path))


def read_gaussians(path: str | Path, sh_degree: int, device: str | torch.device) -> dict:
    vertex = PlyData.read(str(path))["vertex"]

    def columns(names):
        if not names:
            return torch.empty((len(vertex), 0), device=device)
        return torch.tensor(
            np.column_stack([vertex[name] for name in names]), dtype=torch.float32, device=device
        )

    rest = sorted(
        (name for name in vertex.data.dtype.names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if len(rest) != 3 * ((sh_degree + 1) ** 2 - 1):
        raise ValueError("PLY SH degree does not match model configuration")
    return {
        "positions": columns(["x", "y", "z"]),
        "sh_dc": columns([f"f_dc_{i}" for i in range(3)]).reshape(-1, 3, 1).transpose(1, 2),
        "sh_rest": columns(rest).reshape(len(vertex), 3, -1).transpose(1, 2),
        "opacity_logits": columns(["opacity"]),
        "log_scales": columns([f"scale_{i}" for i in range(3)]),
        "rotations": columns([f"rot_{i}" for i in range(4)]),
    }


def write_point_cloud(path: str | Path, cloud: PointCloud) -> None:
    names = ["x", "y", "z", "red", "green", "blue"]
    array = np.empty(
        len(cloud.points), dtype=[(n, "f4" if i < 3 else "u1") for i, n in enumerate(names)]
    )
    values = np.concatenate((cloud.points, np.clip(cloud.colors * 255, 0, 255)), axis=1)
    for index, name in enumerate(names):
        array[name] = values[:, index]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(array, "vertex")]).write(str(path))
