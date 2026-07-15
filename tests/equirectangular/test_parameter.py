from __future__ import annotations

import numpy as np
import pytest
from units import Angle, AngleUnit

from sphere_image.equirectangular import (
    EquirectangularProcessor,
    EquirectangularProcessorParameters,
)


class TestEquirectangularProcessorParameters:
    """Tests for EquirectangularProcessorParameters."""

    def test_default_construction_derives_output_fovs(self) -> None:
        """Default parameters must derive matching output FoVs for a 16:9 canvas."""
        params = EquirectangularProcessorParameters()
        assert params.aspect_ratio == pytest.approx(1280 / 720)
        assert params.output_vfov.degree == pytest.approx(90.0)
        assert params.output_hfov.degree > params.output_vfov.degree

    def test_build_processor_returns_processor_bound_to_image(self) -> None:
        """build_processor must return a processor referencing the given image and params."""
        params = EquirectangularProcessorParameters()
        image = np.zeros((8, 16, 3), dtype=np.uint8)
        processor = params.build_processor(image=image)
        assert isinstance(processor, EquirectangularProcessor)
        assert processor.image is image
        assert processor.params is params

    def test_rejects_output_fov_with_multiple_elements(self) -> None:
        """output_fov with more than one element must raise ValueError."""
        with pytest.raises(ValueError, match="output_fov must contain exactly one element"):
            EquirectangularProcessorParameters(
                output_fov=Angle(value=np.array([10.0, 20.0]), unit=AngleUnit.DEGREE),
            )

    @pytest.mark.parametrize("output_fov_degree", [0.0, 180.0, 270.0])
    def test_rejects_output_fov_outside_valid_range(self, output_fov_degree: float) -> None:
        """output_fov outside (0, 180) degrees must raise ValueError."""
        with pytest.raises(ValueError, match="output_fov must satisfy"):
            EquirectangularProcessorParameters(
                output_fov=Angle(value=np.array([output_fov_degree]), unit=AngleUnit.DEGREE),
            )

    @pytest.mark.parametrize(("output_image_w", "output_image_h"), [(0, 720), (1280, 0), (-1, 720)])
    def test_rejects_non_positive_output_image_size(self, output_image_w: int, output_image_h: int) -> None:
        """Non-positive output_image_w or output_image_h must raise ValueError."""
        with pytest.raises(ValueError, match="must be greater than zero"):
            EquirectangularProcessorParameters(
                output_image_w=output_image_w,
                output_image_h=output_image_h,
            )
