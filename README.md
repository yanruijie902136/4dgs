# 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

## CVPR 2024

### [Project Page](https://guanjunwu.github.io/4dgs/index.html)| [arXiv Paper](https://arxiv.org/abs/2310.08528)

[Guanjun Wu](https://guanjunwu.github.io/) <sup>1*</sup>, [Taoran Yi](https://github.com/taoranyi) <sup>2*</sup>,
[Jiemin Fang](https://jaminfong.cn/) <sup>3‡</sup>, [Lingxi Xie](http://lingxixie.com/) <sup>3 </sup>, </br>[Xiaopeng Zhang](https://scholar.google.com/citations?user=Ud6aBAcAAAAJ&hl=zh-CN) <sup>3 </sup>, [Wei Wei](https://www.eric-weiwei.com/) <sup>1 </sup>,[Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/) <sup>2 </sup>, [Qi Tian](https://www.qitian1987.com/) <sup>3 </sup> , [Xinggang Wang](https://xwcv.github.io) <sup>2‡✉</sup>

<sup>1 </sup>School of CS, HUST &emsp; <sup>2 </sup>School of EIC, HUST &emsp; <sup>3 </sup>Huawei Inc. &emsp;

<sup>\*</sup> Equal Contributions. <sup>$\ddagger$</sup> Project Lead. <sup>✉</sup> Corresponding Author.



![block](assets/teaserfig.jpg)
Our method converges very quickly and achieves real-time rendering speed.

## Refactored implementation

This fork organizes the CVPR 2024 implementation into an installable Python
package with typed Hydra configuration, lazy dataset readers, explicit training
state, and streaming rendering/evaluation. The 4DGS deformation and splatting
method is retained. See [architecture](docs/architecture.md),
[dataset inputs](docs/datasets.md), [migration](docs/migration.md), and
[validation results](docs/validation.md).

## Installation

Linux, an NVIDIA GPU, and a CUDA compiler are required for training/rendering.
Configuration, dataset utilities, and CPU tests do not require CUDA extensions.
The supported environment uses Python 3.11, PyTorch 2.7.1, and CUDA 12.6.

```bash
git submodule update --init --recursive
conda env update -n 4dgs -f environment.yml
conda activate 4dgs
python -m pip install -r requirements.txt
bash scripts/build_extensions.sh
```

For a CPU-only development installation:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

COLMAP is an optional external dependency for reconstruction. The preprocessing
commands target CUDA-enabled COLMAP 3.13; select its executable through
`preprocess.colmap`. Training on already prepared data does not invoke COLMAP.

```bash
conda install -n 4dgs -c conda-forge 'colmap=3.13.0=cuda_126*'
```

`requirements-lock.txt` records the validated Python dependencies, including
optional tools. Use it in place of `requirements.txt` to reproduce that set.
Native builds apply the checked-in CUDA compatibility patches to temporary
copies, preserving the pinned submodules.

## Train, render, evaluate

Dataset roots are always supplied by the caller. Paths below are examples
relative to the launch directory, not assumed installation locations.

```bash
4dgs train experiment=dnerf/bouncingballs \
  data.root=./datasets/bouncingballs run.output_dir=./outputs/bouncingballs

4dgs render experiment=dnerf/bouncingballs \
  data.root=./datasets/bouncingballs \
  checkpoint.path=./outputs/bouncingballs/checkpoints/latest.pt \
  run.output_dir=./outputs/bouncingballs/rendered

4dgs evaluate run.output_dir=./outputs/bouncingballs/rendered
```

All commands also work as `python -m fourdgs <command>`. Use
`4dgs train --help` to inspect configuration, and `--cfg job` to print a composed
experiment without training. Experiment presets live in
[`src/fourdgs/configs/experiment`](src/fourdgs/configs/experiment).

For Neu3D/DyNeRF, the original layout is `camXX/images/*.png` alongside
`poses_bounds.npy`. Alternative prepared layouts are explicit overrides:

```bash
4dgs train experiment=dynerf/coffee_martini \
  data.root=./datasets/coffee_martini \
  'data.images=rgb/2x/{camera}' \
  data.point_cloud=./prepared/points3D_downsample2.ply \
  run.output_dir=./outputs/coffee_martini
```

`data.images` and `data.point_cloud` are relative to `data.root` unless absolute.
Readers support D-NeRF, Neu3D/DyNeRF, HyperNeRF/Nerfies, COLMAP, Panoptic Sports,
and custom multiview metadata. They never extract videos or modify datasets.
`experiment=panoptic/default` retains the original Panoptic renderer's black
background and degree-zero SH while using the shared camera/rendering interfaces.

Resume a complete training checkpoint with the original experiment settings:

```bash
4dgs train experiment=dnerf/bouncingballs data.root=./datasets/bouncingballs \
  run.output_dir=./outputs/bouncingballs \
  checkpoint.path=./outputs/bouncingballs/checkpoints/latest.pt checkpoint.resume=true
```

## Additional workflows

```bash
# Explicit video extraction into a separate destination.
4dgs preprocess data.root=./datasets/coffee_martini \
  preprocess.operation=extract preprocess.output_dir=./prepared/coffee_martini \
  data.width=1352 data.height=1014

# Export deformed 3DGS PLY snapshots at requested times.
4dgs export experiment=dnerf/bouncingballs data.root=./datasets/bouncingballs \
  checkpoint.path=./outputs/bouncingballs/checkpoints/latest.pt \
  run.output_dir=./exports/bouncingballs 'export.times=[0,0.5,1]'

# Run training, rendering, and metric evaluation together.
4dgs benchmark experiment=dnerf/bouncingballs data.root=./datasets/bouncingballs \
  run.output_dir=./outputs/benchmark
```

See [migration](docs/migration.md) for preprocessing with known poses, legacy
checkpoint import, and output locations. Viewer support is opt-in through
`viewer.enabled=true`; the protocol is described in [viewer usage](docs/viewer_usage.md).
The notebooks use user-supplied paths and the same public package functions.

## Development and validation

```bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy
pytest -m 'not cuda and not integration'
pytest -m cuda
```

Tests generate small temporary datasets. Real benchmark data is supplied to
commands separately and is never checked into the repository. The regression
suite covers preset values, camera geometry, lazy readers, optimizer topology,
checkpoint continuation, streaming output, and native forward/backward paths.

## Citation

```bibtex
@inproceedings{wu20244dgaussians,
  title={4D Gaussian Splatting for Real-Time Dynamic Scene Rendering},
  author={Wu, Guanjun and Yi, Taoran and Fang, Jiemin and Xie, Lingxi and Zhang, Xiaopeng and Wei, Wei and Liu, Wenyu and Tian, Qi and Wang, Xinggang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

Upstream authorship and license notices are retained. Consult `LICENSE.md` and
native submodule licenses for the applicable terms.
