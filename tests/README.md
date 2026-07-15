# tests

## Overview

Unit tests for the `sphere_image` package, run with `pytest`. Shared fixtures (synthetic equirectangular/fisheye images and an identity rotation matrix) live in `conftest.py`; test modules mirror the `src/sphere_image` package layout one-to-one.

## Components

| Component | Description |
| --- | --- |
| [conftest.py](./conftest.py) | Shared pytest fixtures: synthetic test images and an identity `RotationMatrix`. |
| [utils/](./utils/) | Tests for `OutputFovBasis` and `SphericalCoordinates`. |
| [equirectangular/](./equirectangular/) | Tests for the equirectangular projection method, parameters, and processor. |
| [fisheye/](./fisheye/) | Tests for the fisheye projection method, parameters, and processor. |

## Examples

Run the full suite from the package root:

```bash
uv run pytest
```
