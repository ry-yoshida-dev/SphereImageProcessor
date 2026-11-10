# SphereImageProcessor

## Overview

SphereImageProcessor (`sphere_image`) is a Python package that remaps fisheye and equirectangular images to other views. It builds per-pixel sampling coordinates with NumPy from the output field of view and a rotation, and warps with `cv2.remap`. Field of view, output resolution and projection method are configured through parameter dataclasses.

| Input | Parameters | Projection methods |
| --- | --- | --- |
| Fisheye | `FisheyeProcessorParameters` | Equidistant, orthographic, stereographic, equisolid |
| Equirectangular | `EquirectangularProcessorParameters` | Perspective (the other `EquirectangularProjectionMethod` values are not implemented and raise `ValueError`) |

For fisheye details, see [src/sphere_image/fisheye/README.md](src/sphere_image/fisheye/README.md).

## Installation

From the package root (the directory containing `pyproject.toml`):

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e .
```

Dependencies are installed automatically. To install dependencies only:

```bash
pip install -r requirements.txt
```

## Example

```python
import cv2
from rotation import RotationMatrix

from sphere_image.fisheye import (
    FisheyeProjectionMethod,
    FisheyeProcessorParameters,
)

image = cv2.imread("input.jpg")
params = FisheyeProcessorParameters(method=FisheyeProjectionMethod.EQUIDISTANT)
processor = params.build_processor(image=image)
out = processor.run_pipeline(rotation_matrix=RotationMatrix.unit_matrix())
```

CLI entrypoints:

```bash
python -m src.sphere_image.fisheye --help
python -m src.sphere_image.equirectangular --help
```

## Interactive viewer

An interactive Streamlit viewer lives under [tools/viewer/](tools/viewer/).
It lets you pick a fisheye or equirectangular sample image (or upload your
own) and switch projection method. The image is converted into an
equirectangular texture and rendered in an embedded WebGL panorama viewer:
dragging to look around and scrolling to zoom run entirely client-side, with
no reload. It is a standalone application built on top of this package, not
part of the installed `sphere_image` library.

```bash
uv sync
uv run streamlit run main.py
```

## Testing

Tests live under [tests/](tests/) and mirror the `src/sphere_image` layout. Install the `dev` dependency group and run:

```bash
uv sync
uv run pytest
```

## Continuous integration

[.github/workflows/ci.yaml](.github/workflows/ci.yaml) runs the same checks on pull requests and pushes to `main`:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
