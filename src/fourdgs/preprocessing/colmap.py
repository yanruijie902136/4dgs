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

"""COLMAP subprocess orchestration and camera/database mapping for known poses."""

import shutil
import sqlite3
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np

from fourdgs.data.registry import load_scene
from fourdgs.io.colmap import rotmat2qvec


def run_colmap(executable: str, command: str, **options) -> None:
    """Pass paths as individual arguments and propagate command failures."""
    arguments = [executable, command]
    for name, value in options.items():
        arguments.extend((f"--{name}", str(value)))
    subprocess.run(arguments, check=True)


def prepare_known_poses(config, output: Path):
    """Select reconstruction views without modifying the original dataset."""
    from fourdgs.rendering.outputs import save_image

    data = deepcopy(config.data)
    data.eval = False
    scene = load_scene(data, load_points=False)
    indices = list(range(len(scene.train)))
    if config.data.format in ("neu3d", "dynerf", "multiview"):
        indices = [i for i in indices if scene.train.records[i].camera.time == 0]
    else:
        stride = max(1, int(np.ceil(len(indices) / 200)))
        indices = indices[::stride]
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    cameras = {}
    for index in indices:
        sample = scene.train[index]
        if sample.image is None:
            raise ValueError("Known-pose reconstruction requires source images")
        name = f"view_{index:06d}.png"
        save_image(images / name, sample.image)
        cameras[name] = sample.camera
    return cameras


def write_known_model(database: Path, destination: Path, cameras: dict) -> None:
    """Use database-assigned image/camera IDs; never assume extraction order."""
    destination.mkdir(parents=True, exist_ok=True)
    camera_lines, image_lines = [], []
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT image_id, name, camera_id FROM images").fetchall()
        for image_id, name, camera_id in rows:
            camera = cameras[name]
            parameters = np.array([camera.fx, camera.fy, camera.cx, camera.cy], dtype=np.float64)
            connection.execute(
                "UPDATE cameras SET model=1, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
                (camera.width, camera.height, parameters.tobytes(), camera_id),
            )
            camera_lines.append(
                f"{camera_id} PINHOLE {camera.width} {camera.height} "
                + " ".join(map(str, parameters))
            )
            matrix = camera.world_to_camera.numpy()
            pose = [*rotmat2qvec(matrix[:3, :3]), *matrix[:3, 3]]
            image_lines.append(f"{image_id} " + " ".join(map(str, pose)) + f" {camera_id} {name}\n")
    (destination / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (destination / "images.txt").write_text("\n".join(image_lines) + "\n")
    (destination / "points3D.txt").write_text("")


def reconstruct(config, *, known_poses: bool) -> Path:
    executable = config.preprocess.colmap
    if shutil.which(executable) is None:
        raise FileNotFoundError(f"COLMAP executable not found: {executable}")
    output = Path(config.preprocess.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    database = output / "database.db"
    if database.exists():
        raise FileExistsError(f"Choose a fresh preprocessing output directory: {database}")
    cameras = prepare_known_poses(config, output) if known_poses else None
    images = output / "images"
    if not known_poses:
        source = Path(config.data.root) / (config.data.images or "input")
        shutil.copytree(source, images, dirs_exist_ok=True)
    run_colmap(
        executable,
        "feature_extractor",
        database_path=database,
        image_path=images,
        **{
            "ImageReader.camera_model": "PINHOLE"
            if known_poses
            else config.preprocess.camera_model,
            "ImageReader.single_camera": 0 if known_poses else 1,
            "ImageReader.single_camera_per_image": int(known_poses),
            "FeatureExtraction.use_gpu": int(config.preprocess.use_gpu),
            "FeatureExtraction.num_threads": config.run.cpu_threads,
        },
    )
    sparse = output / "sparse"
    sparse.mkdir(exist_ok=True)
    if known_poses:
        assert cameras is not None
        write_known_model(database, output / "known", cameras)
    run_colmap(
        executable,
        "exhaustive_matcher",
        database_path=database,
        **{
            "FeatureMatching.use_gpu": int(config.preprocess.use_gpu),
            "FeatureMatching.num_threads": config.run.cpu_threads,
        },
    )
    if known_poses:
        sparse = sparse / "0"
        sparse.mkdir(exist_ok=True)
        run_colmap(
            executable,
            "point_triangulator",
            database_path=database,
            image_path=images,
            input_path=output / "known",
            output_path=sparse,
        )
    else:
        run_colmap(
            executable, "mapper", database_path=database, image_path=images, output_path=sparse
        )
        sparse = sparse / "0"
    dense = output / "dense"
    run_colmap(
        executable,
        "image_undistorter",
        image_path=images,
        input_path=sparse,
        output_path=dense,
        output_type="COLMAP",
    )
    if not config.preprocess.dense:
        return dense
    run_colmap(executable, "patch_match_stereo", workspace_path=dense)
    run_colmap(executable, "stereo_fusion", workspace_path=dense, output_path=dense / "fused.ply")
    return dense / "fused.ply"
