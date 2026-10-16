import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from fourdgs.config import register_configs, validate
from fourdgs.data.registry import load_scene
from fourdgs.data.types import Camera


def test_required_root_and_unknown_keys():
    register_configs()
    with initialize_config_module(config_module="fourdgs.configs", version_base="1.3"):
        config = compose(config_name="config", overrides=["experiment=dnerf/bouncingballs"])
        assert OmegaConf.is_missing(config.data, "root")
        assert config.training.iterations == 20000
        assert config.model.kplanes_config.resolution == [64, 64, 64, 75]
        with pytest.raises(Exception, match="Could not override 'training.typo'"):
            compose(config_name="config", overrides=["training.typo=3"])


def test_every_original_preset():
    expected = json.loads((Path(__file__).parent / "fixtures/legacy_presets.json").read_text())
    register_configs()
    with initialize_config_module(config_module="fourdgs.configs", version_base="1.3"):
        for name, original in expected.items():
            config = compose(
                config_name="config", overrides=[f"experiment={name.removesuffix('.py')}"]
            )
            for group in ("model", "optimization", "loss", "density", "training"):
                for key, value in OmegaConf.to_container(config[group]).items():
                    if key in original and key != "custom_sampler":
                        assert value == original[key], (name, key)


def test_panoptic_preserves_original_render_defaults():
    register_configs()
    with initialize_config_module(config_module="fourdgs.configs", version_base="1.3"):
        config = compose(config_name="config", overrides=["experiment=panoptic/default"])
    assert config.data.format == "panoptic"
    assert config.data.width == config.data.height == 0
    assert not config.data.white_background
    assert config.render.max_sh_degree == 0


def test_dataset_lazy_readonly_and_relative_root(config, dnerf_root, monkeypatch):
    monkeypatch.chdir(dnerf_root.parent)
    config.data.root = dnerf_root.name
    validate(config)
    before = sorted(p.relative_to(dnerf_root) for p in dnerf_root.rglob("*"))
    scene = load_scene(config.data)
    assert len(scene.train) == 4 and len(scene.test) == 2
    assert not scene.train._cache
    sample = scene.train[0]
    assert sample.image.shape == (3, 12, 16)
    assert sample.image.device.type == "cpu"
    assert torch.allclose(sample.image, torch.full_like(sample.image, 128 / 255))
    assert sample.camera.time == 0
    assert scene.train[3].camera.time == 1
    assert sorted(p.relative_to(dnerf_root) for p in dnerf_root.rglob("*")) == before
    scene.train.cache_size = 1
    scene.train._cache.clear()
    for index in range(len(scene.train)):
        scene.train[index]
    assert len(scene.train._cache) == 1


def test_camera_projection_and_center():
    matrix = torch.eye(4)
    matrix[0, 3] = -2
    camera = Camera(matrix, 640, 480, 500, 450, 320, 240, 0, "camera")
    view, projection, center = camera.projection("cpu")
    assert torch.equal(center, torch.tensor([2.0, 0, 0]))
    point = torch.tensor([2.0, 0, 5, 1]) @ projection
    assert torch.allclose(point[:2] / point[3], torch.zeros(2))
    assert torch.equal(view, matrix.T)


def test_zero_timeline_and_empty_eval(config, dnerf_root):
    for split in ("train", "test"):
        path = dnerf_root / f"transforms_{split}.json"
        data = json.loads(path.read_text())
        for frame in data["frames"]:
            frame["time"] = 0
        path.write_text(json.dumps(data))
    config.data.root = str(dnerf_root)
    config.data.eval = False
    scene = load_scene(config.data)
    assert len(scene.test) == 0 and len(scene.train) == 6
    assert all(np.isfinite(record.camera.time) for record in scene.train.records)
