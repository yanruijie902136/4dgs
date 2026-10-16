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

"""Neu3D/DyNeRF readers for explicitly extracted, camera-major image folders."""

import math
from pathlib import Path

import numpy as np

from fourdgs.config import DataConfig
from fourdgs.data.readers.common import (
    camera_from_rt,
    data_path,
    image_size,
    read_point_cloud,
    scene_data,
)
from fourdgs.data.types import FrameRecord, SceneData
from fourdgs.geometry.trajectories import get_spiral


def load(config: DataConfig, *, load_points: bool = True) -> SceneData:
    root = Path(config.root)
    pose_bounds = np.load(root / "poses_bounds.npy")
    original = pose_bounds[:, :-2].reshape(-1, 3, 5)
    poses = np.concatenate([original[..., 1:2], -original[..., :1], original[..., 2:4]], -1)
    # Neu3D camera IDs may have gaps; poses follow sorted video names upstream.
    names = [path.stem for path in sorted(root.glob("cam*.mp4"))]
    if not names:
        names = [f"cam{index:02d}" for index in range(len(poses))]
    if len(names) != len(poses):
        raise ValueError("The number of camera videos does not match poses_bounds.npy")
    train: list[FrameRecord] = []
    test: list[FrameRecord] = []
    first_size = None
    for index, (name, pose) in enumerate(zip(names, poses, strict=True)):
        folder = data_path(
            config, config.images.format(camera=name) if config.images else f"{name}/images"
        )
        paths = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not paths:
            raise FileNotFoundError(f"No extracted frames in {folder}; run '4dgs preprocess'")
        paths = paths[: config.max_frames] if config.max_frames else paths
        width, height = image_size(paths[0], config)
        first_size = width, height
        fov_x = 2 * math.atan(original[index, 1, 4] / (2 * original[index, 2, 4]))
        fov_y = 2 * math.atan(original[index, 0, 4] / (2 * original[index, 2, 4]))
        rotation = -pose[:3, :3].copy()
        rotation[:, 0] *= -1
        translation = -pose[:3, 3] @ rotation
        records = test if index == config.eval_camera else train
        for frame, path in enumerate(paths):
            camera = camera_from_rt(
                rotation,
                translation,
                width,
                height,
                fov_x,
                fov_y,
                frame / (config.max_frames or len(paths)),
                f"{name}_{path.stem}",
            )
            records.append(FrameRecord(camera, path))
    if first_size is None:
        raise ValueError("Neu3D dataset contains no cameras")
    width, height = first_size
    video = []
    for index, pose in enumerate(get_spiral(poses, pose_bounds[:, -2:], n_views=300)):
        rotation = -pose[:3, :3].copy()
        rotation[:, 0] *= -1
        camera = camera_from_rt(
            rotation,
            -pose[:3, 3] @ rotation,
            width,
            height,
            fov_x,
            fov_y,
            index / 300,
            f"spiral_{index:05d}",
        )
        video.append(FrameRecord(camera))
    cloud = (
        read_point_cloud(data_path(config, config.point_cloud or "points3D_downsample2.ply"))
        if load_points
        else None
    )
    return scene_data(config, train, test, video, cloud)
