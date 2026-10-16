"""Exercise real-data train/resume/render/export/compose workflows at small resolution."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
from hydra import compose, initialize_config_module

from fourdgs.config import CompositionItem, register_configs
from fourdgs.data.registry import load_scene
from fourdgs.evaluation.metrics import evaluate
from fourdgs.io.checkpoint import model_from_checkpoint, read_checkpoint
from fourdgs.io.export import export_frames
from fourdgs.rendering.composition import compose_scenes
from fourdgs.rendering.outputs import render_dataset
from fourdgs.rendering.rasterizer import render
from fourdgs.training.engine import train


def check(arguments) -> None:
    register_configs()
    with initialize_config_module(config_module="fourdgs.configs", version_base="1.3"):
        overrides = [f"experiment={arguments.preset}"] if arguments.preset else []
        config = compose(config_name="config", overrides=overrides)
    config.data.root = str(arguments.source.resolve())
    config.data.format = arguments.format
    config.data.images = arguments.images
    config.data.point_cloud = arguments.point_cloud
    if arguments.sparse:
        config.data.sparse = arguments.sparse
    config.data.width, config.data.height = 128, 96
    config.data.cache_size = 8
    config.training.coarse_iterations, config.training.iterations = 3, 4
    config.training.log_interval = 1
    config.training.test_iterations = [3]
    config.training.save_iterations = [2]
    config.training.num_workers = 0
    config.model.kplanes_config.resolution = [8, 8, 8, 8]
    config.model.multires = [1]
    config.model.net_width = 16
    config.density.densify_from_iter = 0
    config.density.densification_interval = 2
    config.density.pruning_from_iter = 0
    config.density.pruning_interval = 2
    config.density.opacity_reset_interval = 3
    config.density.prune_min_points = 0
    config.density.densify_until_iter = 6
    config.run.output_dir = str(arguments.output / "uninterrupted")
    config.perceptual_metrics = False
    final = train(deepcopy(config))
    partial = deepcopy(config)
    partial.run.output_dir = str(arguments.output / "resumed")
    partial.training.stop_after = 2
    checkpoint = train(partial)
    before_resume = read_checkpoint(checkpoint)
    partial.training.stop_after = 0
    partial.checkpoint.path = str(checkpoint)
    partial.checkpoint.resume = True
    resumed = train(partial)
    expected, actual = read_checkpoint(final), read_checkpoint(resumed)
    assert expected["sampler"]["order"] == actual["sampler"]["order"]
    assert expected["sampler"]["cursor"] == actual["sampler"]["cursor"]
    assert torch.equal(expected["rng"]["cuda"][0], actual["rng"]["cuda"][0])
    differences = {}
    for name, value in expected["model"].items():
        assert value.shape == actual["model"][name].shape, name
        differences[name] = (
            float((value - actual["model"][name]).abs().max()) if value.numel() else 0
        )
    # Native atomic reductions are nondeterministic. Adam's tiny epsilon can
    # amplify near-zero gradient differences even before the interrupted run.
    before = read_checkpoint(arguments.output / "uninterrupted/checkpoints/coarse_000002.pt")
    before_difference = max(
        float((before["model"][key] - value).abs().max())
        for key, value in before_resume["model"].items()
        if value.numel()
    )
    scene = load_scene(config.data, load_points=False)
    models = [model_from_checkpoint(payload, config.run.device) for payload in (expected, actual)]
    background = torch.tensor([float(config.data.white_background)] * 3, device=config.run.device)
    image_difference = 0.0
    with torch.no_grad():
        for record in scene.test.records[:2]:
            images = [
                render(record.camera, model, config.render, background).image for model in models
            ]
            torch.testing.assert_close(images[0], images[1], atol=0.01, rtol=0.01)
            image_difference = max(image_difference, float((images[0] - images[1]).abs().max()))
    del models
    config.checkpoint.path = str(final)
    config.render.splits = ["test"]
    config.render.max_frames = 2
    config.render.video = True
    config.run.output_dir = str(arguments.output / "rendered")
    render_dataset(config)
    metrics = evaluate(config)
    config.run.output_dir = str(arguments.output / "exported")
    config.export.times = [0.0, 0.5]
    export_frames(config)
    config.composition = [
        CompositionItem(checkpoint=str(final)),
        CompositionItem(checkpoint=str(final), translation=[0.1, 0, 0], scale=0.5, time_offset=0.1),
    ]
    config.run.output_dir = str(arguments.output / "composed")
    compose_scenes(config)
    (arguments.output / "smoke.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "parameter_max_abs": max(differences.values()),
                "before_resume_parameter_max_abs": before_difference,
                "resume_image_max_abs": image_difference,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", required=True)
    parser.add_argument("--preset", default="")
    parser.add_argument("--images", default="")
    parser.add_argument("--point-cloud", default="")
    parser.add_argument("--sparse", default="")
    check(parser.parse_args())


if __name__ == "__main__":
    main()
