# Refactor validation

Reference source: `b8034bb55aaab6a607e54f6f25d69af66f4ec3f7`.
Validation used Python 3.11, PyTorch 2.7.1 + CUDA 12.6, and one RTX 3090.
Both implementations used the same compiled native extensions and input data.
All dataset locations were supplied at runtime.

## Automated and fixed-state checks

- The latest CPU run passed 39 tests. An earlier native CUDA run passed all
  9 GPU tests, and CUDA Compute Sanitizer reported zero errors. The GPU suite
  was not rerun after the final sampler correction.
- All 23 original presets retain their effective model, optimization, loss,
  density, and training values.
- Fixed-state coarse/fine RGB and depth outputs matched exactly. Position
  gradients differed by at most `3.25e-5`, within the comparison tolerances
  (`atol=1e-4`, `rtol=1e-3`). CUDA reductions are not bitwise deterministic.
- Clone/split/prune parameters and Adam moments matched exactly with fixed
  inputs and RNG state. CPU deformation outputs, regularization, and network
  gradients also matched exactly.
- First-frame comparisons for D-NeRF, Neu3D, and HyperNeRF matched image values,
  focal lengths, and timestamps exactly. Pose/radius differences were below
  `4e-6`.
- Ruff, formatting, type checks across all 52 package modules, wheel/sdist builds,
  and configuration composition from outside the repository pass.

Run synthetic tests with `pytest`; CUDA tests skip when their prerequisites
are unavailable. Run the fixed-state comparison with:

```bash
python scripts/compare_reference.py --reference ./reference-checkout \
  --output ./validation/fixed-state.json
```

## Full training comparison

Runs use seed 0, original effective presets, all coarse/fine iterations, native
image resolution, and complete test splits. Metrics use quantized PNG pairs and
shared original-formula implementations. Higher PSNR/SSIM and lower LPIPS are
better. These are paired validation runs, not estimates across random seeds.

| Scene / implementation | PSNR | SSIM | LPIPS-VGG | Training seconds | Peak GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| bouncingballs / original | 40.81160 | 0.994399 | 0.014927 | 285.1 | 331.3 |
| bouncingballs / refactored | 40.89280 | 0.994376 | 0.015181 | 275.5 | 345.0 |
| coffee_martini / original | 28.63955 | 0.905514 | 0.163249 | 3115.2 | 5151.1 |
| coffee_martini / refactored | 28.06124 | 0.903202 | 0.165573 | 3071.6 | 5071.4 |

Bouncingballs satisfies the planned maximum regressions of 0.2 dB PSNR,
0.005 SSIM, and 0.01 LPIPS. Time covers training and scheduled reporting/saves,
excluding dataset/model construction and final image evaluation. GPU memory is
PyTorch's peak allocated memory; it excludes allocations owned by native CUDA
libraries and the display server. Timings are single-run observations.

Both full training comparisons completed. The coffee_martini result above uses
the corrected seeded DataLoader shuffle sequence and all 300 test frames at
1352 x 1014 resolution. Its PSNR is 0.57832 dB lower than the original, exceeding
the planned 0.2 dB tolerance. SSIM and LPIPS remain within their tolerances.
The cause of the remaining PSNR difference is unresolved; investigation stopped
at the user's request. Full training quality parity is therefore not established.

The refactored coffee_martini run briefly overlapped a small GPU diagnostic,
so its timing is not a strictly isolated measurement. All workflow smoke checks
below passed before the final sampler correction.

Generated checkpoints, renders, benchmark inputs, build artifacts, temporary
logs, and the reference archive were removed at the user's request. This
document retains the results; source datasets and the installed environment
were preserved.

## Reference harness adaptations

`scripts/benchmark_reference.py` imports the archived implementation without
rewriting its model, training loss, densification, or sampler. It disables
optional TensorBoard/socket setup and streams its
own renderer's output instead of retaining all GPU images. It also reinterprets
upstream signed-byte RGB arrays as unsigned bytes for Pillow 12 compatibility;
the image bytes are unchanged. These adaptations are recorded per run.

Both versions use the pinned native sources. The checked-in KNN patch adds the
missing `<cfloat>` include needed by CUDA 12; it does not change the algorithm.

Reproduce a full comparison using explicit paths:

```bash
python scripts/benchmark_reference.py --reference ./reference-checkout \
  --preset ./reference-checkout/arguments/dnerf/bouncingballs.py \
  --source ./prepared/bouncingballs --output ./validation/original/bouncingballs
4dgs benchmark experiment=dnerf/bouncingballs data.root=./prepared/bouncingballs \
  run.output_dir=./validation/new/bouncingballs
```

The original checkout additionally requires `mmcv==1.6.0` and `lpips`; they are
not dependencies of the refactored runtime. Install MMCV with
`pip install --no-build-isolation mmcv==1.6.0 lpips 'setuptools<81'` when running
reference comparisons.

## Workflow coverage and continuation limits

Small real-data checks exercise both stages, clone/split/prune, opacity reset,
checkpoint interruption and continuation, streaming PNG/video, metrics, PLY
export, and two-model composition. They pass for D-NeRF, Neu3D, HyperNeRF,
Panoptic Sports, static COLMAP, and custom multiview inputs. The latter two use
a small reconstruction produced during preprocessing validation.

The CPU optimizer/checkpoint continuation test is bitwise exact. The GPU checks
verify matching sampler state, RNG state, topology shapes, and rendered output
within `atol=0.01, rtol=0.01`. They also measure differences before resumption:
independent CUDA training runs already differ because native atomic reductions
and Adam's small epsilon amplify near-zero gradient differences. Saving full
state enables faithful continuation; it does not make these kernels bitwise
deterministic. No cross-platform bitwise guarantee is implied.

COLMAP 3.13 with CUDA completed feature extraction, matching, known-pose
triangulation, undistortion, PatchMatch stereo, and fusion on 18 reduced-size
views, producing 29,444 fused points. Estimated-pose reconstruction and
undistortion also completed. Video extraction and cloud downsampling passed.
Use `preprocess.dense=false` to stop after reconstruction and undistortion.

The original bouncingballs snapshot was imported and rendered. One checked PNG
matched exactly; the other differed in three color channels by one uint8 level.
The weight-visualization notebook executed, and the training notebook is
validated with reduced iteration/frame limits.
