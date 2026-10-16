"""Small generated fixtures: no machine-specific paths or downloaded datasets."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from fourdgs.config import default_config
from fourdgs.models.gaussian import GaussianModel


@pytest.fixture
def config():
    config = default_config()
    config.model.kplanes_config.resolution = [4, 4, 4, 4]
    config.model.kplanes_config.output_coordinate_dim = 2
    config.model.multires = [1]
    config.model.net_width = 8
    config.model.defor_depth = 0
    config.run.device = "cpu"
    return config


@pytest.fixture
def model(config):
    model = GaussianModel(config.model)
    count, coefficients = 6, (config.model.sh_degree + 1) ** 2
    model.replace_points(
        {
            "positions": torch.rand(count, 3),
            "sh_dc": torch.rand(count, 1, 3),
            "sh_rest": torch.rand(count, coefficients - 1, 3),
            "log_scales": torch.full((count, 3), -3.0),
            "rotations": torch.tensor([[1.0, 0, 0, 0]]).repeat(count, 1),
            "opacity_logits": torch.zeros(count, 1),
        }
    )
    return model


@pytest.fixture
def dnerf_root(tmp_path):
    root = tmp_path / "dataset with spaces"
    root.mkdir()
    for split, count in (("train", 4), ("test", 2)):
        frames = []
        (root / split).mkdir()
        for index in range(count):
            image = np.full((12, 16, 4), 128, dtype=np.uint8)
            image[:, :, 3] = 255
            Image.fromarray(image).save(root / split / f"{index}.png")
            pose = np.eye(4)
            pose[:3, 3] = [index * 0.1, 0, 3]
            frames.append(
                {
                    "file_path": f"{split}/{index}",
                    "time": index / max(count - 1, 1),
                    "transform_matrix": pose.tolist(),
                }
            )
        (root / f"transforms_{split}.json").write_text(
            json.dumps({"camera_angle_x": 0.8, "frames": frames})
        )
    return root
