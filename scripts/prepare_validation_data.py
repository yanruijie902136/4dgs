"""Prepare original-layout benchmark inputs from explicit source paths.

Generated point clouds and frame links are placed in a separate workspace.
The source scenes are never modified. This helper is for reference comparisons;
the new readers also accept explicit image-directory and point-cloud overrides.
"""

import argparse
import shutil
from pathlib import Path


def prepare(source: Path, output: Path, format: str, images: Path | None, cloud: Path | None):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if format == "dnerf":
        for path in source.iterdir():
            (output / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
        return
    if format == "neu3d":
        shutil.copy2(source / "poses_bounds.npy", output / "poses_bounds.npy")
        for path in sorted(source.glob("cam*.mp4")):
            (output / path.name).symlink_to(path.resolve())
            folder = output / path.stem
            folder.mkdir()
            (folder / "images").symlink_to((images / path.stem).resolve(), target_is_directory=True)
    else:
        for path in source.iterdir():
            if path.name != "points3D_downsample2.ply":
                (output / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    if cloud is not None:
        import open3d as o3d

        points = o3d.io.read_point_cloud(str(cloud))
        voxel = 0.02
        while len(points.points) > 40000:
            points = points.voxel_down_sample(voxel)
            voxel += 0.01
        o3d.io.write_point_cloud(str(output / "points3D_downsample2.ply"), points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=["dnerf", "neu3d", "hypernerf"], required=True)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--cloud", type=Path)
    arguments = parser.parse_args()
    prepare(arguments.source, arguments.output, arguments.format, arguments.images, arguments.cloud)


if __name__ == "__main__":
    main()
