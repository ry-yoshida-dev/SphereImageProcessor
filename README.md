# FisheyeImage

FisheyeImage is a Python package for remapping sphere_image images toward a rectilinear-style view. It uses an equidistant projection model, builds per-pixel sampling coordinates with NumPy, and warps with `cv2.remap`. Field of view and output resolution are configured through `FisheyeProcessorParameters`.

For module-level detail, see [src/sphere_image/README.md](src/sphere_image/README.md).

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

CLI entrypoint:

```bash
python -m src.sphere_image.fisheye --help
```

## Testing

Tests live under [tests/](tests/) and mirror the `src/sphere_image` layout. Install the `dev` dependency group and run:

```bash
uv sync
uv run pytest
```
