from __future__ import annotations

import cv2
import numpy as np
import pytest
from rotation import RotationMatrix

from sphere_image.equirectangular import (
    EquirectangularProcessor,
    EquirectangularProcessorParameters,
    EquirectangularProjectionMethod,
)


class TestEquirectangularProcessorDirectionVectorGrid:
    """Tests for EquirectangularProcessor._create_direction_vector_grid."""

    def test_returns_unit_vectors_for_every_output_pixel(
        self, sample_equirectangular_image: np.ndarray
    ) -> None:
        """Every generated direction vector must have unit length."""
        params = EquirectangularProcessorParameters(output_image_w=8, output_image_h=4)
        processor = EquirectangularProcessor(image=sample_equirectangular_image, params=params)
        direction_vectors = processor._create_direction_vector_grid()
        assert direction_vectors.shape == (8 * 4, 3)
        norms = np.linalg.norm(direction_vectors, axis=-1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-10)


class TestEquirectangularProcessorMapRotationToUv:
    """Tests for EquirectangularProcessor._map_rotation_to_uv."""

    def test_rejects_unimplemented_projection_method(
        self, sample_equirectangular_image: np.ndarray, identity_rotation_matrix: RotationMatrix
    ) -> None:
        """A non-perspective method must raise ValueError."""
        params = EquirectangularProcessorParameters(
            method=EquirectangularProjectionMethod.ORTHOGRAPHIC,
            output_image_w=8,
            output_image_h=4,
        )
        processor = EquirectangularProcessor(image=sample_equirectangular_image, params=params)
        with pytest.raises(ValueError, match="Not implemented error"):
            processor._map_rotation_to_uv(rotation_matrix=identity_rotation_matrix)

    def test_forward_direction_maps_to_image_center(
        self, sample_equirectangular_image: np.ndarray, identity_rotation_matrix: RotationMatrix
    ) -> None:
        """With an identity rotation, the center pixel must sample u=v=0.5."""
        params = EquirectangularProcessorParameters(output_image_w=9, output_image_h=5)
        processor = EquirectangularProcessor(image=sample_equirectangular_image, params=params)
        u_coordinates, v_coordinates = processor._map_rotation_to_uv(
            rotation_matrix=identity_rotation_matrix
        )
        center_row, center_col = 2, 4
        assert u_coordinates[center_row, center_col] == pytest.approx(0.5, abs=1e-9)
        assert v_coordinates[center_row, center_col] == pytest.approx(0.5, abs=1e-9)


class TestEquirectangularProcessorRemap:
    """Tests for EquirectangularProcessor.remap."""

    def test_returns_image_shaped_like_uv_grid(self, sample_equirectangular_image: np.ndarray) -> None:
        """The remapped output must match the (h, w) shape of the uv grid."""
        params = EquirectangularProcessorParameters(output_image_w=6, output_image_h=4)
        processor = EquirectangularProcessor(image=sample_equirectangular_image, params=params)
        u_coordinates, v_coordinates = np.meshgrid(
            np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 4)
        )
        remapped_image = processor.remap(u_coordinates=u_coordinates, v_coordinates=v_coordinates)
        assert remapped_image.shape == (4, 6, 3)
        assert remapped_image.dtype == sample_equirectangular_image.dtype


class TestEquirectangularProcessorRunPipeline:
    """Tests for EquirectangularProcessor.run_pipeline."""

    def test_produces_output_image_with_configured_size(
        self, sample_equirectangular_image: np.ndarray, identity_rotation_matrix: RotationMatrix
    ) -> None:
        """The output image must have the configured (h, w) shape."""
        params = EquirectangularProcessorParameters(output_image_w=12, output_image_h=8)
        processor = EquirectangularProcessor(image=sample_equirectangular_image, params=params)
        output_image = processor.run_pipeline(rotation_matrix=identity_rotation_matrix)
        assert output_image.shape == (8, 12, 3)


class TestEquirectangularProcessorFromPath:
    """Tests for EquirectangularProcessor.from_path."""

    def test_loads_image_from_disk(self, tmp_path, sample_equirectangular_image: np.ndarray) -> None:
        """from_path must build a processor from the image written at the given path."""
        image_path = tmp_path / "equirectangular.png"
        cv2.imwrite(str(image_path), sample_equirectangular_image)
        processor = EquirectangularProcessor.from_path(path=str(image_path))
        assert processor.image.shape == sample_equirectangular_image.shape

    def test_rejects_missing_file(self, tmp_path) -> None:
        """from_path must raise ValueError when the image cannot be read."""
        missing_path = tmp_path / "missing.png"
        with pytest.raises(ValueError, match="Failed to read image"):
            EquirectangularProcessor.from_path(path=str(missing_path))
