"""Build pinned extensions from temporary copies with reviewable compatibility patches."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    patches = root / "scripts" / "native_patches"
    with tempfile.TemporaryDirectory(prefix="fourdgs-native-") as temporary:
        for name in ("simple-knn", "depth-diff-gaussian-rasterization"):
            source = root / "submodules" / name
            if not (source / "setup.py").is_file():
                raise FileNotFoundError(f"Initialize submodules before building: {source}")
            destination = Path(temporary) / name
            shutil.copytree(
                source,
                destination,
                ignore=shutil.ignore_patterns(".git", "build", "*.egg-info", "__pycache__"),
            )
            for patch in sorted(patches.glob(f"{name}-*.patch")):
                subprocess.run(["patch", "-p1", "--input", str(patch)], cwd=destination, check=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-build-isolation", str(destination)],
                check=True,
            )


if __name__ == "__main__":
    main()
