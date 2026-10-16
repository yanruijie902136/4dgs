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

"""Nerfies/HyperNeRF metadata with lazy image and optional mask loading."""

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
from fourdgs.data.types import FrameRecord, SceneData


def load(config: DataConfig, *, load_points: bool = True) -> SceneData:
    root = Path(config.root)
    dataset = json.loads((root / "dataset.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())
    ids = dataset["ids"]
    if dataset["val_ids"]:
        train_ids, test_ids = set(dataset["train_ids"]), set(dataset["val_ids"])
    else:
        indices = np.arange(len(ids))[::4]
        train_ids = {ids[i] for i in indices}
        test_ids = {ids[i] for i in (indices + 2)[:-1] if i < len(ids)}
    max_time = max(metadata[name]["warp_id"] for name in ids) or 1
    train, test, all_records = [], [], []
    for name in ids:
        meta = json.loads((root / "camera" / f"{name}.json").read_text())
        path = data_path(config, config.images or "rgb/2x") / f"{name}.png"
        width, height = image_size(path, config)
        # Preserve the original focal/FOV convention; distortion is already
        # accounted for by prepared Nerfies images.
        original_width, original_height = meta["image_size"]
        fov_x = 2 * math.atan(original_width / (2 * meta["focal_length"]))
        fov_y = 2 * math.atan(original_height / (2 * meta["focal_length"]))
        rotation = np.asarray(meta["orientation"]).T
        translation = -np.asarray(meta["position"]) @ rotation
        camera = camera_from_rt(
            rotation,
            translation,
            width,
            height,
            fov_x,
            fov_y,
            metadata[name]["warp_id"] / max_time,
            name,
        )
        mask_path = root / "covisible" / "2x" / "val" / f"{name}.png"
        record = FrameRecord(
            camera, path, mask_path if name in test_ids and mask_path.exists() else None
        )
        if name in train_ids:
            train.append(record)
        if name in test_ids:
            test.append(record)
        all_records.append(FrameRecord(camera))
    # Preserve the old video's first len(test) camera/time sequence.
    video = all_records[: len(test)]
    cloud = (
        read_point_cloud(data_path(config, config.point_cloud or "points3D_downsample2.ply"))
        if load_points
        else None
    )
    return scene_data(config, train, test, video, cloud)
