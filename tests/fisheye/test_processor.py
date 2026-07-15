from __future__ import annotations

import numpy as np
import pytest
from geometry.planar import PolarCoordinate
from rotation import RotationMatrix
from units import Angle, AngleUnit

from sphere_image.fisheye import FisheyeProcessor, FisheyeProcessorParameters


class TestFisheyeProcessorDirectionVectorGrid:
    """Tests for FisheyeProcessor._create_direction_vector_grid."""

    def test_returns_unit_vectors_for_every_output_pixel(
        self, sample_fisheye_image: np.ndarray
    ) -> None:
        """Every generated direction vector must have unit length."""
        params = FisheyeProcessorParameters(output_image_w=8, output_image_h=4)
        processor = FisheyeProcessor(image=sample_fisheye_image, params=params)
        direction_vectors = processor._create_direction_vector_grid()
        assert direction_vectors.value.shape == (8 * 4, 3)
        norms = np.linalg.norm(direction_vectors.value, axis=-1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-10)


class TestFisheyeProcessorPolarCoordinateConversion:
    """Tests for FisheyeProcessor._PolarCoordinate2NormalizedCartesianCoordinate."""

    def test_zero_radius_maps_to_image_center(self, sample_fisheye_image: np.ndarray) -> None:
        """A zero-radius polar coordinate must map to the normalized center (0.5, 0.5)."""
        params = FisheyeProcessorParameters(output_image_w=1, output_image_h=1)
        processor = FisheyeProcessor(image=sample_fisheye_image, params=params)
        polar_coordinate = PolarCoordinate(
            radius=np.array([0.0]),
            angle=Angle(value=np.array([0.0]), unit=AngleUnit.RADIAN),
        )
        u_coordinates, v_coordinates = processor._PolarCoordinate2NormalizedCartesianCoordinate(
            polar_coordinate=polar_coordinate
        )
        assert u_coordinates[0, 0] == pytest.approx(0.5)
        assert v_coordinates[0, 0] == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("angle_radian", "expected_u", "expected_v"),
        [
            (0.0, 1.0, 0.5),
            (np.pi / 2, 0.5, 1.0),
            (np.pi, 0.0, 0.5),
            (-np.pi / 2, 0.5, 0.0),
        ],
    )
    def test_unit_radius_maps_to_expected_rim_position(
        self,
        sample_fisheye_image: np.ndarray,
        angle_radian: float,
        expected_u: float,
        expected_v: float,
    ) -> None:
        """A unit-radius polar coordinate must map to the expected point on the unit rim."""
        params = FisheyeProcessorParameters(output_image_w=1, output_image_h=1)
        processor = FisheyeProcessor(image=sample_fisheye_image, params=params)
        polar_coordinate = PolarCoordinate(
            radius=np.array([1.0]),
            angle=Angle(value=np.array([angle_radian]), unit=AngleUnit.RADIAN),
        )
        u_coordinates, v_coordinates = processor._PolarCoordinate2NormalizedCartesianCoordinate(
            polar_coordinate=polar_coordinate
        )
        assert u_coordinates[0, 0] == pytest.approx(expected_u, abs=1e-9)
        assert v_coordinates[0, 0] == pytest.approx(expected_v, abs=1e-9)


class TestFisheyeProcessorRemap:
    """Tests for FisheyeProcessor.remap."""

    def test_returns_image_shaped_like_uv_grid(self, sample_fisheye_image: np.ndarray) -> None:
        """The remapped output must match the (h, w) shape of the uv grid."""
        params = FisheyeProcessorParameters(output_image_w=6, output_image_h=4)
        processor = FisheyeProcessor(image=sample_fisheye_image, params=params)
        u_coordinates, v_coordinates = np.meshgrid(
            np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 4)
        )
        remapped_image = processor.remap(u_coordinates=u_coordinates, v_coordinates=v_coordinates)
        assert remapped_image.shape == (4, 6, 3)
        assert remapped_image.dtype == sample_fisheye_image.dtype


class TestFisheyeProcessorRunPipeline:
    """Tests for FisheyeProcessor.run_pipeline."""

    def test_produces_output_image_with_configured_size(
        self, sample_fisheye_image: np.ndarray, identity_rotation_matrix: RotationMatrix
    ) -> None:
        """The output image must have the configured (h, w) shape."""
        params = FisheyeProcessorParameters(output_image_w=8, output_image_h=8)
        processor = FisheyeProcessor(image=sample_fisheye_image, params=params)
        output_image = processor.run_pipeline(rotation_matrix=identity_rotation_matrix)
        assert output_image.shape == (8, 8, 3)
