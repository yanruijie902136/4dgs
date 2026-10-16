"""Train an archived implementation, then stream its own renderer through shared metrics.

Only runtime adaptations are applied: disable optional TensorBoard/socket setup,
accept legacy RGB bytes in modern Pillow, and stream output instead of retaining
every rendered GPU frame. Model, loss,
densification and sampling logic remain those of the supplied reference checkout.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fourdgs.config import default_config
from fourdgs.evaluation.metrics import evaluate
from fourdgs.rendering.outputs import save_image


def benchmark(reference: Path, preset: Path, source: Path, output: Path) -> None:
    sys.path.insert(0, str(reference.resolve()))
    spec = importlib.util.spec_from_file_location("reference_train", reference / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from mmcv import Config
    from utils.params_utils import merge_hparams

    original_fromarray = Image.fromarray

    def legacy_fromarray(values, mode=None):
        # Pillow 12 rejects the upstream signed-byte RGB array; reinterpret its
        # identical bytes as uint8, as older Pillow did with mode="RGB".
        if mode == "RGB" and values.dtype == np.int8:
            values = values.view(np.uint8)
        return original_fromarray(values, mode)

    Image.fromarray = legacy_fromarray

    parser = argparse.ArgumentParser()
    data = module.ModelParams(parser)
    optimization = module.OptimizationParams(parser)
    pipeline = module.PipelineParams(parser)
    hidden = module.ModelHiddenParams(parser)
    arguments = parser.parse_args([])
    arguments = merge_hparams(arguments, Config.fromfile(str(preset.resolve())))
    arguments.source_path = str(source.resolve())
    arguments.model_path = str(output.resolve())
    module.args = arguments
    module.TENSORBOARD_FOUND = False
    module.setup_seed(0)
    torch.set_num_threads(8)
    output.mkdir(parents=True, exist_ok=True)
    (output / "adaptations.json").write_text(
        json.dumps(
            {
                "reference": str(reference.resolve()),
                "preset": str(preset.resolve()),
                "tensorboard": False,
                "dataloader_workers": "unchanged from reference",
                "pillow_signed_rgb": "reinterpret int8 bytes as uint8",
                "evaluation": "original renderer, streamed PNG pairs, shared original-formula metrics",
            },
            indent=2,
        )
    )
    model = module.GaussianModel(arguments.sh_degree, hidden.extract(arguments))
    scene = module.Scene(data.extract(arguments), model)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    timer = module.Timer()
    timer.start()
    for stage, steps in (("coarse", arguments.coarse_iterations), ("fine", arguments.iterations)):
        module.scene_reconstruction(
            data.extract(arguments),
            optimization.extract(arguments),
            hidden.extract(arguments),
            pipeline.extract(arguments),
            [3000, 7000, 14000],
            [steps],
            [],
            None,
            -1,
            model,
            scene,
            stage,
            None,
            steps,
            timer,
        )
    torch.cuda.synchronize()
    (output / "resources.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.perf_counter() - start,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            },
            indent=2,
        )
    )
    background = torch.tensor([float(arguments.white_background)] * 3, device="cuda")
    destination = output / "evaluation"
    with torch.no_grad():
        for index, camera in enumerate(scene.getTestCameras()):
            rendered = module.render(
                camera,
                model,
                pipeline.extract(arguments),
                background,
                stage="fine",
                cam_type=scene.dataset_type,
            )["render"]
            name = f"{index:05d}.png"
            save_image(destination / "test/renders" / name, rendered)
            save_image(destination / "test/gt" / name, camera.original_image[:3])
    config = default_config()
    config.run.output_dir = str(destination)
    config.render.splits = ["test"]
    evaluate(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "preset", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    arguments = parser.parse_args()
    benchmark(arguments.reference, arguments.preset, arguments.source, arguments.output)


if __name__ == "__main__":
    main()
