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

"""Reader helpers for paths, camera construction, and scene normalization."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fourdgs.config import DataConfig
from fourdgs.data.types import Camera, FrameDataset, FrameRecord, PointCloud, SceneData


def data_path(config: DataConfig, path: str) -> Path:
    """Resolve an explicitly configured path relative to the dataset root."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else Path(config.root).expanduser() / candidate


def image_size(path: Path, config: DataConfig) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if config.width and config.height:
        return config.width, config.height
    return max(1, round(width / config.downsample)), max(1, round(height / config.downsample))


def camera_from_rt(rotation, translation, width, height, fov_x, fov_y, time, name) -> Camera:
    """Convert the original transposed-rotation representation into the contract."""
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.asarray(rotation).T
    matrix[:3, 3] = translation
    return Camera(
        torch.from_numpy(matrix),
        int(width),
        int(height),
        width / (2 * math.tan(fov_x / 2)),
        height / (2 * math.tan(fov_y / 2)),
        width / 2,
        height / 2,
        float(time),
        str(name),
    )


def scene_data(
    config: DataConfig,
    train: list[FrameRecord],
    test: list[FrameRecord],
    video: list[FrameRecord],
    cloud: PointCloud | None,
) -> SceneData:
    if not config.eval:
        train, test = train + test, []
    if not train:
        raise ValueError("Dataset has no training frames")
    centers = np.stack([r.camera.center.numpy() for r in train])
    radius = float(np.linalg.norm(centers - centers.mean(0), axis=-1).max() * 1.1)
    if radius <= 0:
        radius = 1.0
    return SceneData(
        FrameDataset(train, config.white_background, config.cache_size),
        FrameDataset(test, config.white_background, config.cache_size),
        FrameDataset(video, config.white_background, config.cache_size),
        cloud,
        radius,
        config.format,
    )


def read_point_cloud(path: Path) -> PointCloud:
    from plyfile import PlyData

    vertices = PlyData.read(str(path))["vertex"]
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(np.float32)
    colors = (
        np.column_stack([vertices[name] for name in ("red", "green", "blue")]).astype(np.float32)
        / 255
    )
    if not len(points) or not np.isfinite(points).all():
        raise ValueError(f"Point cloud is empty or nonfinite: {path}")
    return PointCloud(points, colors)
