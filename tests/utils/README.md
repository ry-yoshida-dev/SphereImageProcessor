# utils

## Overview

Tests for the shared geometry helpers in `sphere_image.utils`: FoV-basis derivation and spherical/equirectangular coordinate conversion.

## Components

| Component | Description |
| --- | --- |
| [test_basis.py](./test_basis.py) | Tests `OutputFovBasis.build_output_fovs` (vertical/horizontal basis, square aspect ratio, pinhole tangent relationship). |
| [test_spherical_coordinates.py](./test_spherical_coordinates.py) | Tests `SphericalCoordinates.hypotenuse`, `from_cartesian`, `u_coordinates`, and `v_coordinates`. |
