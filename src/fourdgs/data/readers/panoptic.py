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

"""Panoptic Sports pinhole metadata without rasterizer objects in the dataset."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from fourdgs.config import DataConfig
from fourdgs.data.readers.common import data_path, scene_data
from fourdgs.data.types import Camera, FrameRecord, PointCloud, SceneData


def load(config: DataConfig, *, load_points: bool = True) -> SceneData:
    splits = []
    for split in ("train", "test"):
        meta = json.loads((Path(config.root) / f"{split}_meta.json").read_text())
        records = []
        width = config.width or round(meta["w"] / config.downsample)
        height = config.height or round(meta["h"] / config.downsample)
        sx, sy = width / meta["w"], height / meta["h"]
        for time, filenames in enumerate(meta["fn"]):
            for k, matrix, filename in zip(
                meta["k"][time], meta["w2c"][time], filenames, strict=True
            ):
                camera = Camera(
                    torch.tensor(matrix, dtype=torch.float32),
                    width,
                    height,
                    k[0][0] * sx,
                    k[1][1] * sy,
                    k[0][2] * sx,
                    k[1][2] * sy,
                    time / len(meta["fn"]),
                    filename.replace("/", "_"),
                )
                records.append(
                    FrameRecord(camera, data_path(config, config.images or "ims") / filename)
                )
        splits.append(records)
    cloud = None
    if load_points:
        points = np.load(data_path(config, config.point_cloud or "init_pt_cld.npz"))["data"]
        cloud = PointCloud(points[:, :3], points[:, 3:6])
    return scene_data(
        config, splits[0], splits[1], [replace(r, image_path=None) for r in splits[1]], cloud
    )
