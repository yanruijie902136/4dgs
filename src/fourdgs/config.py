"""Typed experiment configuration, independent of CLI and GPU initialization."""

from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class DataConfig:
    """Locations are supplied by the caller; child paths are relative to root."""

    root: str = MISSING
    format: str = "dnerf"
    images: str = ""
    point_cloud: str = ""
    sparse: str = "sparse/0"
    white_background: bool = True
    eval: bool = True
    downsample: float = 1.0
    width: int = 0
    height: int = 0
    holdout: int = 8
    eval_camera: int = 0
    max_frames: int = 300
    extension: str = ".png"
    add_points: bool = False
    cache_size: int = 32


@dataclass
class GridConfig:
    grid_dimensions: int = 2
    input_coordinate_dim: int = 4
    output_coordinate_dim: int = 32
    resolution: list[int] = field(default_factory=lambda: [64, 64, 64, 25])


@dataclass
class ModelConfig:
    """4DGS architecture; original ablation names remain recognizable."""

    sh_degree: int = 3
    net_width: int = 64
    defor_depth: int = 1
    bounds: float = 1.6
    kplanes_config: GridConfig = field(default_factory=GridConfig)
    multires: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    no_dx: bool = False
    no_grid: bool = False
    no_ds: bool = False
    no_dr: bool = False
    no_do: bool = True
    no_dshs: bool = True
    empty_voxel: bool = False
    static_mlp: bool = False
    apply_rotation: bool = False
    grid_pe: int = 0
    # Retained for exact initialization and import of original model weights.
    timebase_pe: int = 4
    posebase_pe: int = 10
    scale_rotation_pe: int = 2
    opacity_pe: int = 2
    timenet_width: int = 64
    timenet_output: int = 32


@dataclass
class LossConfig:
    lambda_dssim: float = 0.0
    plane_tv_weight: float = 0.0001
    time_smoothness_weight: float = 0.01
    l1_time_planes: float = 0.0001


@dataclass
class OptimizationConfig:
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 20000
    deformation_lr_init: float = 0.00016
    deformation_lr_final: float = 0.000016
    deformation_lr_delay_mult: float = 0.01
    grid_lr_init: float = 0.0016
    grid_lr_final: float = 0.00016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001


@dataclass
class DensityConfig:
    percent_dense: float = 0.01
    opacity_reset_interval: int = 3000
    densification_interval: int = 100
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold_coarse: float = 0.0002
    densify_grad_threshold_fine_init: float = 0.0002
    densify_grad_threshold_after: float = 0.0002
    pruning_from_iter: int = 500
    pruning_interval: int = 100
    opacity_threshold_coarse: float = 0.005
    opacity_threshold_fine_init: float = 0.005
    opacity_threshold_fine_after: float = 0.005
    max_points: int = 360000
    prune_min_points: int = 200000


@dataclass
class TrainingConfig:
    iterations: int = 30000
    coarse_iterations: int = 3000
    batch_size: int = 1
    dataloader: bool = False
    num_workers: int = 4
    zerostamp_init: bool = False
    custom_sampler: bool = False
    test_iterations: list[int] = field(default_factory=lambda: [3000, 7000, 14000])
    save_iterations: list[int] = field(default_factory=lambda: [14000, 20000, 30000, 45000, 60000])
    checkpoint_interval: int = 0
    log_interval: int = 10
    render_process: bool = False
    detect_anomaly: bool = False
    # A resumable early stop for integration checks; stage lengths stay unchanged.
    stop_after: int = 0


@dataclass
class RenderConfig:
    max_sh_degree: int = 3
    convert_sh_python: bool = False
    compute_covariance_python: bool = False
    debug: bool = False
    splits: list[str] = field(default_factory=lambda: ["test", "video"])
    video: bool = True
    fps: int = 30
    max_frames: int = 0


@dataclass
class ViewerConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 6009


@dataclass
class RunConfig:
    output_dir: str = "outputs/default"
    seed: int = 0
    device: str = "cuda"
    tensorboard: bool = False
    cpu_threads: int = 8


@dataclass
class CheckpointConfig:
    path: str = ""
    resume: bool = False


@dataclass
class PreprocessConfig:
    operation: str = "extract"
    output_dir: str = MISSING
    colmap: str = "colmap"
    input: str = ""
    max_points: int = 40000
    voxel_size: float = 0.02
    use_gpu: bool = True
    camera_model: str = "OPENCV"
    dense: bool = True


@dataclass
class ExportConfig:
    times: list[float] = field(default_factory=list)
    split: str = "test"


@dataclass
class CompositionItem:
    checkpoint: str = MISSING
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    scale: float = 1.0
    time_offset: float = 0.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    run: RunConfig = field(default_factory=RunConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    composition: list[CompositionItem] = field(default_factory=list)
    perceptual_metrics: bool = True


def register_configs() -> None:
    """Register the schema once before Hydra composes a command."""
    ConfigStore.instance().store(name="schema", node=Config)


def validate(config: DictConfig, *, needs_data: bool = True) -> None:
    """Validate cross-field constraints before allocating models or loading data."""
    if needs_data:
        root = Path(config.data.root).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        config.data.root = str(root.resolve())
    if config.training.batch_size < 1 or config.training.num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if config.training.iterations < 1 or config.training.coarse_iterations < 0:
        raise ValueError("iterations must be positive and coarse_iterations nonnegative")
    if config.data.downsample <= 0:
        raise ValueError("data.downsample must be positive")
    if config.run.cpu_threads < 1 or config.training.log_interval < 1:
        raise ValueError("cpu_threads and log_interval must be positive")
    if config.data.holdout < 2:
        raise ValueError("data.holdout must be at least two")
    if config.data.max_frames < 0 or config.render.max_frames < 0:
        raise ValueError("Frame limits must be nonnegative")
    if (config.data.width > 0) != (config.data.height > 0):
        raise ValueError("Set both data.width and data.height")
    if not 0 <= config.model.sh_degree <= 3:
        raise ValueError("The rasterizer supports SH degrees 0 through 3")
    if not 0 <= config.render.max_sh_degree <= 3:
        raise ValueError("render.max_sh_degree must be between zero and three")
    for name in ("densification_interval", "pruning_interval", "opacity_reset_interval"):
        if config.density[name] <= 0:
            raise ValueError(f"density.{name} must be positive")


def default_config() -> DictConfig:
    """Construct a structured configuration for direct library use."""
    return OmegaConf.structured(Config)
