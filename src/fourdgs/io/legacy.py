"""One-way import of original 4DGS weights into the versioned checkpoint format."""

import re
from pathlib import Path

import numpy as np
import torch
from numpy.core.multiarray import scalar as numpy_scalar  # type: ignore[attr-defined]

from fourdgs.io.checkpoint import save_checkpoint
from fourdgs.io.ply import read_gaussians
from fourdgs.models.gaussian import GaussianModel


def import_legacy(config) -> Path:
    """Import PLY/deformation directories or original positional checkpoint tuples.

    Legacy optimizer/sampler state is deliberately not represented as an exact
    continuation: the original format never recorded the complete random state.
    """
    source = Path(config.checkpoint.path)
    model = GaussianModel(config.model)
    stage, step = "fine", 0
    if source.is_dir():
        model.replace_points(
            read_gaussians(source / "point_cloud.ply", config.model.sh_degree, "cpu")
        )
        weights = torch.load(source / "deformation.pth", map_location="cpu", weights_only=True)
        model.deformation.load_state_dict(weights)
        model.active_sh_degree = config.model.sh_degree
        iteration = re.fullmatch(r"(coarse_)?iteration_(\d+)", source.name)
        if iteration:
            step = int(iteration[2])
            if iteration[1]:
                stage = "coarse"
                model.active_sh_degree = min(config.model.sh_degree, step // 1000)
    else:
        # Original camera normalization stores the LR scale as a NumPy scalar.
        # Permit only its scalar/dtype constructors in the weights-only loader.
        scalar_types = [
            numpy_scalar,
            np.dtype,
            type(np.dtype("float32")),
            type(np.dtype("float64")),
        ]
        with torch.serialization.safe_globals(scalar_types):
            parameters, step = torch.load(source, map_location="cpu", weights_only=True)
        if len(parameters) != 14:
            raise ValueError("Expected the original 14-field 4DGS checkpoint tuple")
        degree, xyz, deformation, _, dc, rest, scales, rotations, opacity, *_ = parameters
        model.replace_points(
            dict(
                positions=xyz,
                sh_dc=dc,
                sh_rest=rest,
                log_scales=scales,
                rotations=rotations,
                opacity_logits=opacity,
            )
        )
        model.deformation.load_state_dict(deformation)
        model.active_sh_degree = degree
        model.spatial_lr_scale = float(parameters[-1])
        stage = "coarse" if "coarse" in source.name else "fine"
    output = Path(config.run.output_dir) / "imported.pt"
    save_checkpoint(output, model, config, stage=stage, step=step)
    return output
