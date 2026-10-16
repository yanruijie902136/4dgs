# Dataset inputs

Pass `data.root` on the command line or in your own configuration. There is no
default dataset location. The paths below describe files *within that root*;
they are format conventions, not locations on the host machine. Image loading
is lazy and CPU-only, and readers do not create or modify dataset files.

| `data.format` | Required metadata | Default images | Initial points |
| --- | --- | --- | --- |
| `dnerf` | `transforms_train.json`, `transforms_test.json` | Each frame's `file_path` | `fused.ply`, or 2,000 random points when absent |
| `neu3d` / `dynerf` | `poses_bounds.npy` | `camXX/images/*` | `points3D_downsample2.ply` |
| `hypernerf` / `nerfies` | `dataset.json`, `metadata.json`, `camera/<id>.json` | `rgb/2x/<id>.png` | `points3D_downsample2.ply` |
| `colmap` | `sparse/0/cameras.{bin,txt}`, `sparse/0/images.{bin,txt}` | `images/<COLMAP image name>` | `sparse/0/points3D.ply`, falling back to `.bin` or `.txt` |
| `panoptic` | `train_meta.json`, `test_meta.json` | `ims/<metadata filename>` | `init_pt_cld.npz`, array `data` containing XYZ and RGB |
| `multiview` | COLMAP cameras/images and `poses_bounds_multipleview.npy` | `camXX/*` | `points3D_multipleview.ply` |

PLY point clouds contain `x`, `y`, `z`, `red`, `green`, and `blue` vertex
properties; colors are uint8 values. Panoptic NPZ colors are already in `[0, 1]`.
Initial points are not needed when loading a checkpoint for rendering or
continuation.

## Path and resolution overrides

`data.images`, `data.point_cloud`, and `data.sparse` resolve relative to
`data.root`; absolute paths are accepted. D-NeRF image paths come directly from
its transforms files. For Neu3D, `{camera}` expands to a name such as `cam04`.
For multiview, it expands to the two-digit camera number, such as `04`.

```bash
4dgs train experiment=dynerf/coffee_martini \
  data.root=./datasets/coffee_martini \
  'data.images=rgb/2x/{camera}' \
  data.point_cloud=cloud/initial.ply \
  run.output_dir=./outputs/coffee_martini
```

Set both `data.width` and `data.height` for a fixed output resolution. Otherwise,
`data.downsample` divides the dimensions of the prepared images. Some experiment
presets set a fixed resolution; use `data.width=0 data.height=0` to return to
image-derived sizing. Intrinsics are scaled with the image dimensions.
`data.cache_size` bounds each dataset's CPU image cache, per worker process;
set it to zero to disable caching.

## Splits and time conventions

- D-NeRF reads the explicit train/test transforms and normalizes timestamps by
  the largest timestamp across both splits. Missing or all-zero times become
  zero. RGBA images are composited against `data.white_background`.
- Neu3D holds out the camera at `data.eval_camera` (default zero). Camera order
  follows sorted `cam*.mp4` names, retaining gaps in camera IDs. If videos are
  absent, names are contiguous `cam00`, `cam01`, and so on. `data.max_frames`
  limits each sequence and supplies its time denominator; zero loads all frames
  and uses the sequence length. The default is 300 frames.
- HyperNeRF uses `train_ids` and `val_ids`. With no validation IDs, it retains
  the original every-fourth training frame and offset evaluation selection.
  Time is `warp_id` divided by the maximum warp ID. Optional covisibility masks
  under `covisible/2x/val` are exposed with frame samples.
- Static COLMAP sorts image names and holds out every `data.holdout` image.
  All timestamps are zero. Cameras must already use `PINHOLE` or
  `SIMPLE_PINHOLE`; run undistortion during preprocessing for other models.
- Panoptic uses its explicit metadata splits. Time is the timestep index divided
  by the number of timesteps. `experiment=panoptic/default` preserves the original
  black background and degree-zero SH renderer settings.
- Multiview retains the original convention: every frame trains, and frames at
  indices `0`, `N // 3`, and `2 * N // 3` also form the evaluation sequence.
  These evaluation frames are therefore not held out. Time is the frame index
  divided by the sequence length.

`data.eval=false` includes held-out frames in training and leaves the test
split empty. Multiview frames already in training are not duplicated.

Video extraction, cloud downsampling, and COLMAP reconstruction are explicit
`4dgs preprocess` operations with caller-supplied output directories. See
[migration examples](migration.md#preprocessing-and-composition-examples).
