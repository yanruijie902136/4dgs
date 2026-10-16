"""Run the public training/rendering/evaluation pipeline for one experiment."""

from copy import deepcopy
from pathlib import Path

from fourdgs.evaluation.metrics import evaluate
from fourdgs.rendering.outputs import render_dataset
from fourdgs.training.engine import train


def benchmark(config) -> Path:
    """Hydra multirun can compose this operation across experiment presets."""
    checkpoint = train(config)
    render_config = deepcopy(config)
    render_config.checkpoint.path = str(checkpoint.resolve())
    render_config.checkpoint.resume = False
    render_config.render.splits = ["test"]
    render_config.render.video = False
    render_config.run.output_dir = str(Path(config.run.output_dir) / "evaluation")
    output = render_dataset(render_config)
    evaluate(render_config)
    return output
