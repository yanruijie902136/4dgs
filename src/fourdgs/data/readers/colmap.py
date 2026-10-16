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

"""Static COLMAP and custom multiview datasets sharing the same camera parser."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from fourdgs.config import DataConfig
from fourdgs.data.readers.common import data_path, image_size, read_point_cloud, scene_data
from fourdgs.data.types import Camera, FrameRecord, PointCloud, SceneData
from fourdgs.io import colmap as io


def read_reconstruction(path: Path):
    if (path / "images.bin").exists():
        return io.read_extrinsics_binary(path / "images.bin"), io.read_intrinsics_binary(
            path / "cameras.bin"
        )
    return io.read_extrinsics_text(path / "images.txt"), io.read_intrinsics_text(
        path / "cameras.txt"
    )


def make_camera(extrinsic, intrinsic, width, height, time, name):
    if intrinsic.model == "SIMPLE_PINHOLE":
        fx, cx, cy = intrinsic.params
        fy = fx
    elif intrinsic.model == "PINHOLE":
        fx, fy, cx, cy = intrinsic.params
    else:
        raise ValueError(f"Undistort {intrinsic.model} cameras before training")
    sx, sy = width / intrinsic.width, height / intrinsic.height
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = io.qvec2rotmat(extrinsic.qvec)
    matrix[:3, 3] = extrinsic.tvec
    return Camera(
        torch.from_numpy(matrix), width, height, fx * sx, fy * sy, cx * sx, cy * sy, time, name
    )


def load(config: DataConfig, *, load_points: bool = True) -> SceneData:
    sparse = data_path(config, config.sparse)
    extrinsics, intrinsics = read_reconstruction(sparse)
    records = []
    for extrinsic in sorted(extrinsics.values(), key=lambda item: item.name):
        path = data_path(config, config.images or "images") / extrinsic.name
        width, height = image_size(path, config)
        records.append(
            FrameRecord(
                make_camera(
                    extrinsic, intrinsics[extrinsic.camera_id], width, height, 0.0, path.stem
                ),
                path,
            )
        )
    train = [r for i, r in enumerate(records) if not config.eval or i % config.holdout]
    test = [r for i, r in enumerate(records) if config.eval and i % config.holdout == 0]
    cloud = None
    if load_points:
        path = (
            data_path(config, config.point_cloud) if config.point_cloud else sparse / "points3D.ply"
        )
        if path.exists() or config.point_cloud:
            cloud = read_point_cloud(path)
        else:
            reader = (
                io.read_points3d_binary
                if (sparse / "points3D.bin").exists()
                else io.read_points3d_text
            )
            xyz, rgb, _ = reader(
                sparse / ("points3D.bin" if reader == io.read_points3d_binary else "points3D.txt")
            )
            cloud = PointCloud(xyz, rgb / 255)
    return scene_data(config, train, test, [replace(r, image_path=None) for r in train], cloud)


def load_multiview(config: DataConfig, *, load_points: bool = True) -> SceneData:
    """All times train; the original three timestamps per camera form evaluation."""
    import re

    from fourdgs.data.readers.common import camera_from_rt
    from fourdgs.geometry.trajectories import get_spiral

    extrinsics, intrinsics = read_reconstruction(data_path(config, config.sparse))
    train, test = [], []
    for extrinsic in extrinsics.values():
        number = re.search(r"(\d+)(?=\.[^.]+$)", extrinsic.name)
        if number is None:
            raise ValueError(f"Cannot identify multiview camera from {extrinsic.name}")
        folder = data_path(
            config, (config.images or "cam{camera}").format(camera=f"{int(number[0]):02d}")
        )
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
        if not paths:
            raise ValueError(f"No images in {folder}")
        width, height = image_size(paths[0], config)
        for index, path in enumerate(paths):
            camera = make_camera(
                extrinsic,
                intrinsics[extrinsic.camera_id],
                width,
                height,
                index / len(paths),
                f"{folder.name}_{path.stem}",
            )
            record = FrameRecord(camera, path)
            train.append(record)
            if index in (0, len(paths) // 3, len(paths) * 2 // 3):
                test.append(record)
    bounds = np.load(data_path(config, "poses_bounds_multipleview.npy"))
    poses = bounds[:, :-2].reshape(-1, 3, 5)
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    first = train[0].camera
    video = []
    for index, pose in enumerate(get_spiral(poses, bounds[:, -2:], n_views=300)):
        rotation = -pose[:3, :3].copy()
        rotation[:, 0] *= -1
        camera = camera_from_rt(
            rotation,
            -pose[:3, 3] @ rotation,
            first.width,
            first.height,
            2 * np.arctan(first.width / (2 * first.fx)),
            2 * np.arctan(first.height / (2 * first.fy)),
            index / 300,
            f"spiral_{index:05d}",
        )
        video.append(FrameRecord(camera))
    cloud = (
        read_point_cloud(data_path(config, config.point_cloud or "points3D_multipleview.ply"))
        if load_points
        else None
    )
    # Evaluation views already belong to training in this legacy format.
    # Disabling evaluation must not append those same frames a second time.
    return scene_data(config, train, test if config.eval else [], video, cloud)
