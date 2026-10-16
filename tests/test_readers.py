import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fourdgs.data.registry import load_scene


def png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 16, 3), 90, dtype=np.uint8)).save(path)


def test_neu3d_explicit_image_pattern(config, tmp_path):
    bounds = []
    for index, name in enumerate(("cam00", "cam04")):
        (tmp_path / f"{name}.mp4").touch()
        pose = np.column_stack((np.eye(3), np.array([index, 0, 3]), [12, 16, 12]))
        bounds.append(np.r_[pose.flatten(), [0.1, 4]])
        for frame in range(4):
            png(tmp_path / "frames" / name / f"{frame:04d}.png")
    np.save(tmp_path / "poses_bounds.npy", bounds)
    config.data.root = str(tmp_path)
    config.data.format = "neu3d"
    config.data.images = "frames/{camera}"
    config.data.max_frames = 4
    scene = load_scene(config.data, load_points=False)
    assert len(scene.train) == len(scene.test) == 4
    assert scene.test[0].camera.name.startswith("cam00")
    assert scene.train[0].camera.name.startswith("cam04")
    assert scene.train[3].camera.time == 0.75
    assert scene.train[0].camera.fx == pytest.approx(12)
    assert len(scene.video) == 300


def test_hypernerf_split_and_intrinsics(config, tmp_path):
    ids = ["000", "001", "002"]
    (tmp_path / "dataset.json").write_text(
        json.dumps({"ids": ids, "train_ids": ids[:2], "val_ids": ids[2:]})
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({name: {"warp_id": i, "camera_id": i} for i, name in enumerate(ids)})
    )
    (tmp_path / "camera").mkdir()
    for i, name in enumerate(ids):
        (tmp_path / "camera" / f"{name}.json").write_text(
            json.dumps(
                {
                    "image_size": [32, 24],
                    "orientation": np.eye(3).tolist(),
                    "position": [i, 0, 3],
                    "focal_length": 24,
                }
            )
        )
        png(tmp_path / "rgb" / "2x" / f"{name}.png")
    config.data.root = str(tmp_path)
    config.data.format = "hypernerf"
    scene = load_scene(config.data, load_points=False)
    assert len(scene.train) == 2 and len(scene.test) == 1
    assert scene.test[0].camera.time == 1
    assert scene.test[0].camera.fx == pytest.approx(12)


def test_panoptic_off_center_projection(config, tmp_path):
    matrix = np.eye(4).tolist()
    intrinsic = [[12, 0, 6], [0, 11, 5], [0, 0, 1]]
    for split in ("train", "test"):
        meta = {
            "w": 16,
            "h": 12,
            "fn": [["cam0/0.png"]],
            "k": [[intrinsic]],
            "w2c": [[matrix]],
            "cam_id": [[0]],
        }
        (tmp_path / f"{split}_meta.json").write_text(json.dumps(meta))
    png(tmp_path / "ims" / "cam0" / "0.png")
    config.data.root = str(tmp_path)
    config.data.format = "panoptic"
    scene = load_scene(config.data, load_points=False)
    camera = scene.train[0].camera
    assert camera.cx == 6 and camera.cy == 5
    _, projection, _ = camera.projection("cpu")
    assert projection[2, 0].item() == pytest.approx(-0.25)


def colmap_metadata(root: Path, names):
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 16 12 12 11 8 6\n")
    lines = [f"{i + 1} 1 0 0 0 {i} 0 3 1 {name}\n\n" for i, name in enumerate(names)]
    (sparse / "images.txt").write_text("".join(lines))


def test_colmap_text_reader(config, tmp_path):
    names = ["0.png", "1.png", "2.png"]
    colmap_metadata(tmp_path, names)
    for name in names:
        png(tmp_path / "images" / name)
    config.data.root = str(tmp_path)
    config.data.format = "colmap"
    config.data.holdout = 2
    scene = load_scene(config.data, load_points=False)
    assert len(scene.train) == 1 and len(scene.test) == 2
    assert scene.train[0].camera.name == "1"


def test_multiview_reader(config, tmp_path):
    colmap_metadata(tmp_path, ["frame_00001.png", "frame_00002.png"])
    bounds = []
    for index in (1, 2):
        for frame in range(3):
            png(tmp_path / f"cam{index:02d}" / f"frame_{frame + 1:05d}.png")
        pose = np.column_stack((np.eye(3), np.array([index, 0, 3]), [12, 16, 12]))
        bounds.append(np.r_[pose.flatten(), [0.1, 4]])
    np.save(tmp_path / "poses_bounds_multipleview.npy", bounds)
    config.data.root = str(tmp_path)
    config.data.format = "multiview"
    scene = load_scene(config.data, load_points=False)
    assert len(scene.train) == len(scene.test) == 6
    assert len(scene.video) == 300
    config.data.eval = False
    training_only = load_scene(config.data, load_points=False)
    assert len(training_only.train) == 6 and len(training_only.test) == 0
