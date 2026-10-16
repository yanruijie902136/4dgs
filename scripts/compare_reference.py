"""Compare fixed-state forward/backward behavior against an original checkout.

Usage: python scripts/compare_reference.py --reference /path/to/original --output result.json
Both implementations must use the same installed native extensions and PyTorch.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from fourdgs.config import default_config
from fourdgs.data.types import Camera, PointCloud
from fourdgs.models.gaussian import GaussianModel
from fourdgs.rendering.rasterizer import render
from fourdgs.training.density import DensityController
from fourdgs.training.engine import seed_everything
from fourdgs.training.optimizer import OptimizerController

POINT_NAMES = {
    "positions": "_xyz",
    "sh_dc": "_features_dc",
    "sh_rest": "_features_rest",
    "log_scales": "_scaling",
    "rotations": "_rotation",
    "opacity_logits": "_opacity",
}


def compare_topology(old, new, config) -> dict:
    """Compare clone/split/prune and Adam moments with identical inputs and RNG."""
    arguments = SimpleNamespace(
        **OmegaConf.to_container(config.optimization), percent_dense=config.density.percent_dense
    )
    old.training_setup(arguments)
    optimizer = OptimizerController(new, config.optimization)
    density = DensityController(new, optimizer, config.density)
    with torch.no_grad():
        scales = torch.full_like(new.log_scales, 0.1)
        scales[: new.count // 2] = 0.005
        old._scaling.copy_(scales.log())
        new.log_scales.copy_(scales.log())
    old.optimizer.zero_grad(set_to_none=True)
    optimizer.optimizer.zero_grad(set_to_none=True)
    sum(getattr(old, name).square().sum() for name in POINT_NAMES.values()).backward()
    sum(value.square().sum() for value in new.point_parameters().values()).backward()
    old.optimizer.step()
    old.optimizer.zero_grad(set_to_none=True)
    optimizer.step()

    gradients = torch.linspace(0, 0.001, new.count, device=new.device).reshape(-1, 1)
    old.xyz_gradient_accum = gradients.clone()
    old.denom = torch.ones_like(gradients)
    density.gradient_sum = gradients.clone()
    density.observations = torch.ones_like(gradients)
    state = torch.cuda.get_rng_state()
    old.densify(0.00025, 0.005, 1.0, None, 5, 5)
    torch.cuda.set_rng_state(state)
    density.densify(0.00025, 1.0)
    assert len(old._xyz) == new.count
    remove = torch.arange(new.count, device=new.device) % 3 == 0
    old.prune_points(remove)
    density.prune(remove)

    parameter_error = moment_error = 0.0
    for name, original_name in POINT_NAMES.items():
        expected, actual = getattr(old, original_name), getattr(new, name)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        parameter_error = max(parameter_error, float((expected - actual).abs().max().detach()))
        for key in ("step", "exp_avg", "exp_avg_sq"):
            left = old.optimizer.state[expected][key]
            right = optimizer.optimizer.state[actual][key]
            torch.testing.assert_close(left, right, rtol=0, atol=0)
            moment_error = max(moment_error, float((left - right).abs().max()))
    return {
        "topology_parameter_max_abs": parameter_error,
        "topology_adam_moment_max_abs": moment_error,
    }


def compare(reference: Path) -> dict:
    sys.path.insert(0, str(reference.resolve()))
    from gaussian_renderer import render as original_render
    from scene.gaussian_model import GaussianModel as OriginalModel
    from utils.graphics_utils import BasicPointCloud

    config = default_config()
    config.model.multires = [1, 2]
    config.model.kplanes_config.resolution = [8, 8, 8, 8]
    config.model.net_width = 16
    config.model.defor_depth = 0
    config.model.no_do = False
    config.model.no_dshs = False
    arguments = SimpleNamespace(**OmegaConf.to_container(config.model))
    points = np.random.default_rng(13).uniform(-0.5, 0.5, (100, 3)).astype(np.float32)
    points[:, 2] += 3
    colors = np.random.default_rng(17).uniform(0, 1, points.shape).astype(np.float32)
    seed_everything(0)
    old = OriginalModel(3, arguments)
    old._deformation.deformation_net.set_aabb(points.max(0), points.min(0))
    old.create_from_pcd(BasicPointCloud(points, colors, np.zeros_like(points)), 1.0, 1)
    seed_everything(0)
    new = GaussianModel(config.model)
    new.initialize(PointCloud(points, colors), 1.0, "cuda")
    for key, value in old._deformation.state_dict().items():
        torch.testing.assert_close(value, new.deformation.state_dict()[key], rtol=0, atol=0)
    camera = Camera(torch.eye(4), 64, 48, 60, 60, 32, 24, 0.3, "comparison")
    view, projection, center = camera.projection("cuda")
    original_camera = SimpleNamespace(
        FoVx=2 * np.arctan(64 / 120),
        FoVy=2 * np.arctan(48 / 120),
        image_width=64,
        image_height=48,
        time=0.3,
        world_view_transform=view,
        full_proj_transform=projection,
        camera_center=center,
    )
    pipeline = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
    background = torch.zeros(3, device="cuda")
    differences = {}
    for stage in ("coarse", "fine"):
        old._xyz.grad = None
        new.zero_grad(set_to_none=True)
        expected = original_render(original_camera, old, pipeline, background, stage=stage)
        actual = render(camera, new, config.render, background, stage)
        for name, left, right in [
            ("image", actual.image, expected["render"]),
            ("depth", actual.depth, expected["depth"]),
        ]:
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-4)
            differences[f"{stage}_{name}_max_abs"] = float((left - right).abs().max())
        expected["render"].sum().backward()
        actual.image.sum().backward()
        torch.testing.assert_close(old._xyz.grad, new.positions.grad, atol=1e-4, rtol=1e-3)
        differences[f"{stage}_position_gradient_max_abs"] = float(
            (old._xyz.grad - new.positions.grad).abs().max()
        )
    differences.update(compare_topology(old, new, config))
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    results = compare(arguments.reference)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
