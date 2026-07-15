from __future__ import annotations

import numpy as np
import pytest
from rotation import RotationMatrix


@pytest.fixture
def identity_rotation_matrix() -> RotationMatrix:
    """
    Provide an identity rotation matrix shared across processor tests.

    Returns
    -------
    RotationMatrix
        The 3x3 identity rotation matrix.
    """
    return RotationMatrix.unit_matrix()


@pytest.fixture
def sample_equirectangular_image() -> np.ndarray:
    """
    Provide a small synthetic equirectangular BGR image for processor tests.

    Returns
    -------
    np.ndarray
        shape (32, 64, 3), dtype uint8. Horizontal gradient in the blue
        channel and vertical gradient in the green channel.
    """
    height, width = 32, 64
    blue_channel = np.linspace(0, 255, width, dtype=np.uint8)
    green_channel = np.linspace(0, 255, height, dtype=np.uint8)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = blue_channel[np.newaxis, :]
    image[:, :, 1] = green_channel[:, np.newaxis]
    return image


@pytest.fixture
def sample_fisheye_image() -> np.ndarray:
    """
    Provide a small synthetic square fisheye BGR image for processor tests.

    Returns
    -------
    np.ndarray
        shape (64, 64, 3), dtype uint8. Horizontal gradient in the blue
        channel and vertical gradient in the green channel.
    """
    size = 64
    blue_channel = np.linspace(0, 255, size, dtype=np.uint8)
    green_channel = np.linspace(0, 255, size, dtype=np.uint8)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = blue_channel[np.newaxis, :]
    image[:, :, 1] = green_channel[:, np.newaxis]
    return image
