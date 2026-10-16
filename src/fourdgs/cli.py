"""Small command dispatcher; Hydra handles experiment composition and overrides."""

import logging
import sys

import hydra
from omegaconf import DictConfig

from fourdgs.config import register_configs

COMMANDS = (
    "train",
    "render",
    "evaluate",
    "preprocess",
    "export",
    "compose",
    "benchmark",
    "import-legacy",
)


def dispatch(command: str, config: DictConfig):
    """Import heavyweight dependencies only for the requested operation."""
    if command == "train":
        from fourdgs.training.engine import train

        return train(config)
    if command == "render":
        from fourdgs.rendering.outputs import render_dataset

        return render_dataset(config)
    if command == "evaluate":
        from fourdgs.evaluation.metrics import evaluate

        return evaluate(config)
    if command == "preprocess":
        from fourdgs.preprocessing.commands import preprocess

        return preprocess(config)
    if command == "export":
        from fourdgs.io.export import export_frames

        return export_frames(config)
    if command == "compose":
        from fourdgs.rendering.composition import compose_scenes

        return compose_scenes(config)
    if command == "import-legacy":
        from fourdgs.io.legacy import import_legacy

        return import_legacy(config)
    if command == "benchmark":
        from fourdgs.evaluation.benchmark import benchmark

        return benchmark(config)
    raise ValueError(command)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: 4dgs <command> [Hydra overrides]\nCommands: " + ", ".join(COMMANDS))
        print(
            "Example: 4dgs train experiment=dnerf/bouncingballs data.root=./datasets/bouncingballs"
        )
        return
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}; choose from {', '.join(COMMANDS)}")
    register_configs()

    @hydra.main(version_base="1.3", config_path="configs", config_name="config")
    def run(config: DictConfig):
        result = dispatch(command, config)
        logging.getLogger(__name__).info("%s: %s", command, result)

    run()
