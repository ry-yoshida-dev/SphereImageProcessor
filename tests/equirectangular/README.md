# equirectangular

## Overview

Tests for the `sphere_image.equirectangular` package: the projection method enum, parameter validation, and the remapping processor.

## Components

| Component | Description |
| --- | --- |
| [test_method.py](./test_method.py) | Tests `EquirectangularProjectionMethod` member values. |
| [test_parameter.py](./test_parameter.py) | Tests `EquirectangularProcessorParameters` FoV derivation, `build_processor`, and validation errors. |
| [test_processor.py](./test_processor.py) | Tests `EquirectangularProcessor` direction-vector generation, UV mapping, `remap`, `run_pipeline`, and `from_path`. |
