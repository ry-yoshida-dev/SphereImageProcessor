# fisheye

## Overview

Tests for the `sphere_image.fisheye` package: the projection method enum, parameter validation, and the remapping processor.

## Components

| Component | Description |
| --- | --- |
| [test_method.py](./test_method.py) | Tests `FisheyeProjectionMethod.calculate_radius` for every projection law. |
| [test_parameter.py](./test_parameter.py) | Tests `FisheyeProcessorParameters` FoV derivation, `intrinsic_parameter`, `build_processor`, and validation errors. |
| [test_processor.py](./test_processor.py) | Tests `FisheyeProcessor` direction-vector generation, polar-to-Cartesian conversion, `remap`, and `run_pipeline`. |
