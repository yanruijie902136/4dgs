import json
import socket
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from fourdgs.evaluation.metrics import evaluate
from fourdgs.preprocessing.colmap import run_colmap, write_known_model
from fourdgs.rendering.outputs import save_image
from fourdgs.rendering.viewer import Viewer, receive_exact


def test_cli_outside_repository(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fourdgs",
            "train",
            "--cfg",
            "job",
            "experiment=dnerf/bouncingballs",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "iterations: 20000" in result.stdout
    assert not list(tmp_path.iterdir())


def test_subprocess_paths_are_single_arguments():
    with patch("subprocess.run") as run:
        run_colmap("/tools/my colmap", "feature_extractor", image_path="a folder/images")
    run.assert_called_once_with(
        ["/tools/my colmap", "feature_extractor", "--image_path", "a folder/images"], check=True
    )


def test_image_write_failure_propagates(tmp_path):
    with patch.object(Image.Image, "save", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            save_image(tmp_path / "image.png", torch.zeros(3, 12, 16))


def test_streaming_metrics_and_pair_validation(config, tmp_path):
    config.run.output_dir = str(tmp_path)
    config.render.splits = ["test"]
    config.perceptual_metrics = False
    image = torch.full((3, 16, 16), 0.5)
    for index in range(3):
        save_image(tmp_path / "test/renders" / f"{index}.png", image)
        save_image(tmp_path / "test/gt" / f"{index}.png", image + 0.1)
    metrics = evaluate(config)
    assert metrics["test"]["PSNR"] == pytest.approx(20, abs=0.2)
    assert len(json.loads((tmp_path / "per_view.json").read_text())["test"]) == 3
    save_image(tmp_path / "test/renders/extra.png", image)
    with pytest.raises(ValueError, match="names differ"):
        evaluate(config)


def test_viewer_reads_fragmented_messages():
    left, right = socket.socketpair()
    try:
        left.sendall(b"ab")
        left.sendall(b"cd")
        assert receive_exact(right, 4) == b"abcd"
        left.close()
        with pytest.raises(ConnectionError):
            receive_exact(right, 1)
    finally:
        left.close()
        right.close()


def test_viewer_acknowledges_control_only_message(config):
    config.viewer.enabled = False
    viewer = Viewer(config.viewer, "a scene directory")
    left, right = socket.socketpair()
    viewer.listener = object()
    viewer.connection = right
    message = json.dumps(
        {"resolution_x": 0, "resolution_y": 0, "train": True, "keep_alive": False}
    ).encode()
    try:
        left.sendall(len(message).to_bytes(4, "little") + message)
        viewer.service(None, None, None, None, "coarse", True)
        length = int.from_bytes(receive_exact(left, 4), "little")
        assert receive_exact(left, length) == b"a scene directory"
    finally:
        left.close()
        right.close()


def test_known_model_uses_database_ids(tmp_path):
    import sqlite3

    from fourdgs.data.types import Camera

    database = tmp_path / "database.db"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE images (image_id INTEGER, name TEXT, camera_id INTEGER);
            CREATE TABLE cameras (camera_id INTEGER, model INTEGER, width INTEGER,
                                  height INTEGER, params BLOB, prior_focal_length INTEGER);
            INSERT INTO images VALUES (42, 'second.png', 7), (13, 'first.png', 9);
            INSERT INTO cameras VALUES (7, 0, 0, 0, NULL, 0), (9, 0, 0, 0, NULL, 0);
        """)
    camera = Camera(torch.eye(4), 16, 12, 12, 11, 8, 6, 0, "view")
    output = tmp_path / "known"
    write_known_model(database, output, {"first.png": camera, "second.png": camera})
    lines = (output / "images.txt").read_text().splitlines()
    assert lines[0].startswith("42 ") and lines[0].endswith(" 7 second.png")
    assert lines[2].startswith("13 ") and lines[2].endswith(" 9 first.png")


def test_export_with_explicit_times_needs_no_dataset(config, model, tmp_path):
    from fourdgs.io.checkpoint import save_checkpoint
    from fourdgs.io.export import export_frames

    config.checkpoint.path = str(tmp_path / "input.pt")
    config.run.output_dir = str(tmp_path / "export")
    config.export.times = [0.0, 0.5]
    save_checkpoint(config.checkpoint.path, model, config, stage="fine", step=1)
    output = export_frames(config)
    assert len(list(output.glob("*.ply"))) == 2
    assert json.loads((output / "timestamps.json").read_text()) == [0, 0.5]


def test_legacy_coarse_snapshot_preserves_stage(config, model, tmp_path):
    from fourdgs.io.checkpoint import read_checkpoint
    from fourdgs.io.legacy import import_legacy
    from fourdgs.io.ply import write_gaussians

    source = tmp_path / "coarse_iteration_10"
    write_gaussians(source / "point_cloud.ply", model.attributes(0, "coarse"))
    torch.save(model.deformation.state_dict(), source / "deformation.pth")
    config.checkpoint.path = str(source)
    config.run.output_dir = str(tmp_path / "converted")
    imported = read_checkpoint(import_legacy(config))
    assert imported["stage"] == "coarse" and imported["step"] == 10
    assert imported["active_sh_degree"] == 0


def test_render_context_adopts_checkpoint_architecture(config, model, dnerf_root, tmp_path):
    from fourdgs.config import default_config
    from fourdgs.io.checkpoint import save_checkpoint
    from fourdgs.rendering.outputs import load_render_context

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, config, stage="fine", step=1)
    rendering = default_config()
    rendering.data.root = str(dnerf_root)
    rendering.run.device = "cpu"
    rendering.checkpoint.path = str(path)
    restored, _, _, _ = load_render_context(rendering)
    assert rendering.model.net_width == config.model.net_width
    assert restored.config.net_width == config.model.net_width


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
def test_legacy_tuple_with_numpy_scene_radius(config, model, tmp_path, scalar_type):
    from fourdgs.io.checkpoint import read_checkpoint
    from fourdgs.io.legacy import import_legacy

    parameters = (
        1,
        model.positions,
        model.deformation.state_dict(),
        torch.ones(model.count, dtype=torch.bool),
        model.sh_dc,
        model.sh_rest,
        model.log_scales,
        model.rotations,
        model.opacity_logits,
        torch.zeros(model.count),
        torch.zeros(model.count, 1),
        torch.zeros(model.count, 1),
        {},
        scalar_type(1.5),
    )
    source = tmp_path / "chkpnt_coarse_1000.pth"
    torch.save((parameters, 1000), source)
    config.checkpoint.path = str(source)
    config.run.output_dir = str(tmp_path / "converted")
    payload = read_checkpoint(import_legacy(config))
    assert payload["stage"] == "coarse" and payload["step"] == 1000
    assert payload["spatial_lr_scale"] == 1.5
    assert type(payload["spatial_lr_scale"]) is float
    assert torch.equal(payload["model"]["positions"], model.positions)
