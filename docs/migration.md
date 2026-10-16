# Migrating from the original scripts

Activate the new environment and install the package before using these commands.
The old Python import paths and argparse/MMCV configuration files are replaced.

| Previous workflow | New command |
| --- | --- |
| `train.py` | `4dgs train experiment=dnerf/bouncingballs data.root=... run.output_dir=...` |
| `render.py` | `4dgs render experiment=... data.root=... checkpoint.path=... run.output_dir=...` |
| `metrics.py` | `4dgs evaluate run.output_dir=...` |
| `full_eval.py` | `4dgs benchmark experiment=... data.root=... run.output_dir=...` |
| `export_perframe_3DGS.py` | `4dgs export experiment=... data.root=... checkpoint.path=...` |
| `merge_many_4dgs.py` | `4dgs compose` with a configured `composition` list |
| Video extraction scripts | `4dgs preprocess preprocess.operation=extract` |
| `downsample_point.py` | `4dgs preprocess preprocess.operation=downsample preprocess.input=...` |
| `train_test_split.py` | `4dgs preprocess preprocess.operation=split` |
| `convert.py` | `4dgs preprocess preprocess.operation=colmap` |
| `colmap.sh` and pose converters | `4dgs preprocess preprocess.operation=known-poses` |

All preprocessing commands require `data.root` and a separate
`preprocess.output_dir`. Set `preprocess.colmap` to a different executable when
needed. No preprocessing command removes an existing reconstruction directory.

## Configuration

`ModelParams` becomes `data`, with SH degree under `model`. `ModelHiddenParams`
becomes `model` and `loss`. `OptimizationParams` is separated into `training`,
`optimization`, and `density`. `PipelineParams` becomes `render`. All 23 original
experiment presets are migrated; fields which never affected execution are
omitted. `dynerf` presets select the `neu3d` reader, and `dycheck` presets select
the Nerfies metadata reader.

Set booleans explicitly, for example `data.white_background=false`. Unknown
keys are rejected. Viewer connections require `viewer.enabled=true`.

## Checkpoints and output

Training writes a resolved `config.yaml`, JSONL progress, resource measurements,
and `checkpoints/latest.pt` plus named stage snapshots. Resume with the same
experiment settings, `checkpoint.path=...`, and `checkpoint.resume=true`.
For a warm start, leave `checkpoint.resume=false`.

Import an original snapshot directory containing `point_cloud.ply` and
`deformation.pth`, or an original `chkpnt_*.pth`, with:

```bash
4dgs import-legacy experiment=dnerf/bouncingballs \
  checkpoint.path=./old-snapshot run.output_dir=./converted
```

Use the resulting `imported.pt` for rendering or warm starts. The original
format did not save sufficient random/sampling state for exact continuation.
Legacy NumPy scene-radius scalars are converted to plain floats during import.

Rendered frames live in `<output>/<split>/renders`, paired targets in `gt`,
and videos in `<output>/<split>/video.mp4`. Evaluation writes `results.json`
and `per_view.json`. Per-time Gaussian exports live under `gaussians/`.

## Behavior corrections

- Epoch exhaustion fetches a fresh batch instead of reusing the previous one.
- Python covariance and SH/color alternatives supply mutually exclusive native
  arguments, computed from the deformed Gaussian state.
- Per-time export uses deformed opacity.
- Dataset loading is read-only and image caching is bounded.
- Disabling multiview evaluation does not duplicate frames already used for training.
- Render/export write failures propagate; empty evaluation splits are skipped.
- All-zero timestamps remain finite, and non-square camera geometry uses the
  correct height, width, and focal axes.
- Nonfinite training losses raise an error rather than restarting the process.
- Pruning never removes the final Gaussian, and opacity reset also works before
  Adam has initialized its moments.

Exploratory helpers with hard-coded scene paths are replaced by public Python
operations and parameterized commands; obsolete scratch code is not a supported
compatibility interface. Original source remains available in Git history.

## Preprocessing and composition examples

Known camera poses can initialize COLMAP reconstruction without writing inside
the source dataset:

```bash
4dgs preprocess experiment=dynerf/coffee_martini \
  data.root=./datasets/coffee_martini preprocess.operation=known-poses \
  preprocess.output_dir=./prepared/reconstruction
4dgs preprocess data.root=./prepared/reconstruction \
  preprocess.operation=downsample preprocess.input=dense/fused.ply \
  preprocess.output_dir=./prepared/cloud
```

For composition, create `custom-configs/composition.yaml`:

```yaml
# @package _global_
defaults:
  - config
  - _self_
composition:
  - checkpoint: ./outputs/scene-a/checkpoints/latest.pt
    translation: [0, 0, 0]
    rotation: [1, 0, 0, 0]  # wxyz quaternion
    scale: 1.0
    time_offset: 0.0
  - checkpoint: ./outputs/scene-b/checkpoints/latest.pt
    translation: [1, 0, 0]
    rotation: [1, 0, 0, 0]
    scale: 0.5
    time_offset: 0.1
```

```bash
4dgs compose --config-dir ./custom-configs --config-name composition \
  experiment=dnerf/bouncingballs data.root=./datasets/bouncingballs \
  run.output_dir=./outputs/composed
```

The public `compose_scenes(config)` function accepts the same configuration.
The source dataset supplies the output camera
trajectory. Each model keeps its own deformation field and active SH degree;
transformed points are jointly depth-sorted by the rasterizer.
