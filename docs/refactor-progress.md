# Refactor implementation record

Reference: `b8034bb55aaab6a607e54f6f25d69af66f4ec3f7`.
The original source and pinned native submodules were archived outside the
working tree for numerical comparisons; the temporary archive was removed
after validation. Dataset paths are runtime inputs only.

Implemented: an installable `fourdgs` package; typed Hydra configuration with
all 23 original presets and a Panoptic compatibility preset; CPU camera/frame
contracts and six dataset readers; Gaussian/deformation model; rasterizer
adapter; separate sampling, optimization, density control, checkpoints, and
training; streaming rendering/evaluation; preprocessing, legacy import, export,
scene composition, viewer support, CLI, tests, and updated notebooks/docs.

Validation completed: 39 CPU tests in the latest run and 9 native CUDA tests
in an earlier run; CUDA memcheck with zero reported errors; Ruff and type
checks across all 52 package modules;
wheel/sdist and outside-repository CLI checks; fixed-state original/new GPU
comparison; small train/resume/render/evaluate/export/compose runs for all six
reader families; video extraction/downsampling; known-pose COLMAP dense
reconstruction and estimated-pose reconstruction/undistortion; original
snapshot import/render comparison; full bouncingballs and coffee_martini
before/after training. GPU workflow checks predate the final sampler correction.

The coffee_martini comparison has an unresolved 0.57832 dB PSNR regression,
outside the planned 0.2 dB tolerance. Bouncingballs passes the quality tolerances.
Further investigation stopped at the user's request, and generated validation
and build artifacts were removed. Source datasets and the configured environment
remain. See [validation details](validation.md) for results and limitations.
