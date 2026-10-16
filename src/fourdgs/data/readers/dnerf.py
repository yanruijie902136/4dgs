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

"""D-NeRF Blender transforms, including the original orbit trajectory."""

import json
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
from fourdgs.data.types import FrameRecord, PointCloud, SceneData
from fourdgs.geometry.sh import sh_to_rgb


def blender_camera(matrix, width, height, fov_x, time, name):
    inverse = np.linalg.inv(np.asarray(matrix))
    rotation = -inverse[:3, :3].T
    rotation[:, 0] *= -1
    focal = width / (2 * math.tan(fov_x / 2))
    return camera_from_rt(
        rotation,
        -inverse[:3, 3],
        width,
        height,
        fov_x,
        2 * math.atan(height / (2 * focal)),
        time,
        name,
    )


def load(config: DataConfig, *, load_points: bool = True) -> SceneData:
    root = Path(config.root)
    metadata = [
        json.loads((root / f"transforms_{split}.json").read_text()) for split in ("train", "test")
    ]
    max_time = max(float(frame.get("time", 0)) for meta in metadata for frame in meta["frames"])
    splits = []
    for meta in metadata:
        records = []
        fov_x = meta.get("camera_angle_x")
        if fov_x is None:
            fov_x = 2 * math.atan(meta["w"] / (2 * meta["fl_x"]))
        for frame in meta["frames"]:
            path = root / frame["file_path"]
            if not path.suffix:
                path = path.with_suffix(config.extension)
            width, height = image_size(path, config)
            camera = blender_camera(
                frame["transform_matrix"],
                width,
                height,
                fov_x,
                float(frame.get("time", 0)) / (max_time or 1),
                path.stem,
            )
            records.append(FrameRecord(camera, path, composite_alpha=True, resize_filter="bicubic"))
        splits.append(records)
    first = splits[0][0].camera
    video = []
    for index, angle in enumerate(np.linspace(-np.pi, np.pi, 160, endpoint=False)):
        transform = np.eye(4)
        transform[2, 3] = 4
        phi = -np.pi / 6
        rot_x = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(phi), -np.sin(phi), 0],
                [0, np.sin(phi), np.cos(phi), 0],
                [0, 0, 0, 1],
            ]
        )
        rot_y = np.array(
            [
                [np.cos(angle), 0, -np.sin(angle), 0],
                [0, 1, 0, 0],
                [np.sin(angle), 0, np.cos(angle), 0],
                [0, 0, 0, 1],
            ]
        )
        axis = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        matrix = axis @ rot_y @ rot_x @ transform
        video.append(
            FrameRecord(
                blender_camera(
                    matrix,
                    first.width,
                    first.height,
                    2 * math.atan(first.width / (2 * first.fx)),
                    index / 159,
                    f"orbit_{index:05d}",
                )
            )
        )
    cloud = None
    if load_points:
        path = data_path(config, config.point_cloud or "fused.ply")
        if path.exists():
            cloud = read_point_cloud(path)
        elif config.point_cloud:
            raise FileNotFoundError(path)
        else:
            cloud = PointCloud(
                np.random.random((2000, 3)) * 2.6 - 1.3,
                sh_to_rgb(np.random.random((2000, 3)) / 255),
            )
    return scene_data(config, splits[0], splits[1], video, cloud)
