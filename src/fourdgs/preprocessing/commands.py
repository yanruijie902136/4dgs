"""Explicit preprocessing operations; source dataset files are read-only inputs."""

import json
import shutil
from pathlib import Path

from fourdgs.config import validate
from fourdgs.data.readers.common import data_path, read_point_cloud
from fourdgs.io.ply import write_point_cloud


def extract_frames(config) -> Path:
    """Extract video frames into output/camXX/images, reporting decode/write errors."""
    import cv2

    output = Path(config.preprocess.output_dir)
    videos = sorted(Path(config.data.root).glob("cam*.mp4"))
    if config.preprocess.input:
        videos = [data_path(config.data, config.preprocess.input)]
    if not videos:
        raise ValueError("No input videos found")
    for video in videos:
        destination = output / video.stem / "images"
        destination.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        count = 0
        try:
            while not config.data.max_frames or count < config.data.max_frames:
                success, frame = capture.read()
                if not success:
                    break
                height, width = frame.shape[:2]
                size = (
                    (config.data.width, config.data.height)
                    if config.data.width
                    else (
                        round(width / config.data.downsample),
                        round(height / config.data.downsample),
                    )
                )
                if size != (width, height):
                    frame = cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
                if not cv2.imwrite(str(destination / f"{count:04d}.png"), frame):
                    raise OSError(f"Cannot write frame {count} of {video}")
                count += 1
        finally:
            capture.release()
        if not count:
            raise ValueError(f"Video contains no decodable frames: {video}")
    return output


def downsample_cloud(config) -> Path:
    """Use the original increasing voxel-size policy to reach the requested count."""
    import open3d as o3d

    source = data_path(config.data, config.preprocess.input or config.data.point_cloud)
    if not source.is_file():
        raise FileNotFoundError(source)
    if config.preprocess.max_points < 1 or config.preprocess.voxel_size <= 0:
        raise ValueError("max_points and voxel_size must be positive")
    cloud = o3d.io.read_point_cloud(str(source))
    if not len(cloud.points):
        raise ValueError(f"Empty point cloud: {source}")
    voxel = config.preprocess.voxel_size
    while len(cloud.points) > config.preprocess.max_points:
        cloud = cloud.voxel_down_sample(voxel)
        voxel += 0.01
    output = Path(config.preprocess.output_dir) / "points3D_downsample2.ply"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud):
        raise OSError(f"Cannot write {output}")
    return output


def split_transforms(config) -> Path:
    """Create D-NeRF-style train/test folders from a transforms.json sequence."""
    root = Path(config.data.root)
    metadata = json.loads((root / (config.preprocess.input or "transforms.json")).read_text())
    output = Path(config.preprocess.output_dir)
    if config.data.holdout < 2:
        raise ValueError("data.holdout must be at least two")
    splits = {name: {**metadata, "frames": []} for name in ("train", "test")}
    for index, original in enumerate(metadata["frames"]):
        frame = dict(original)
        split = "test" if index % config.data.holdout == 0 else "train"
        source = root / frame["file_path"]
        if not source.suffix:
            source = source.with_suffix(config.data.extension)
        name = f"{index:06d}{source.suffix}"
        destination = output / split / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        frame.update(
            time=frame.get("time", index / len(metadata["frames"])), file_path=f"{split}/{name}"
        )
        splits[split]["frames"].append(frame)
    for split, meta in splits.items():
        (output / f"transforms_{split}.json").write_text(json.dumps(meta, indent=2))
    return output


def preprocess(config) -> Path:
    validate(config)
    operation = config.preprocess.operation
    output = Path(config.preprocess.output_dir).expanduser().resolve()
    source = Path(config.data.root).resolve()
    if output == source:
        raise ValueError("Preprocessing output must be separate from the source dataset root")
    config.preprocess.output_dir = str(output)
    if operation == "extract":
        return extract_frames(config)
    if operation == "downsample":
        return downsample_cloud(config)
    if operation == "split":
        return split_transforms(config)
    if operation in ("colmap", "known-poses"):
        from fourdgs.preprocessing.colmap import reconstruct

        return reconstruct(config, known_poses=operation == "known-poses")
    if operation == "copy-cloud":
        cloud = read_point_cloud(data_path(config.data, config.preprocess.input))
        path = output / "points.ply"
        write_point_cloud(path, cloud)
        return path
    raise ValueError(f"Unknown preprocessing operation: {operation}")
