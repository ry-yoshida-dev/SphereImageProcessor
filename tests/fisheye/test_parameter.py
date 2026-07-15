from __future__ import annotations

import numpy as np
import pytest
from units import Angle, AngleUnit

from sphere_image.fisheye import FisheyeProcessor, FisheyeProcessorParameters


class TestFisheyeProcessorParameters:
    """Tests for FisheyeProcessorParameters."""

    def test_default_construction_derives_output_fovs(self) -> None:
        """Default parameters must derive equal FoVs for the square default canvas."""
        params = FisheyeProcessorParameters()
        assert params.aspect_ratio == pytest.approx(1.0)
        assert params.output_hfov.degree == pytest.approx(params.output_vfov.degree)
        assert params.output_vfov.degree == pytest.approx(90.0)

    def test_build_processor_returns_processor_bound_to_image(self) -> None:
        """build_processor must return a processor referencing the given image and params."""
        params = FisheyeProcessorParameters()
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        processor = params.build_processor(image=image)
        assert isinstance(processor, FisheyeProcessor)
        assert processor.image is image
        assert processor.params is params

    def test_intrinsic_parameter_places_principal_point_at_center(self) -> None:
        """The intrinsic matrix's principal point must sit at the image center."""
        params = FisheyeProcessorParameters(output_image_w=100, output_image_h=200)
        intrinsic_parameter = params.intrinsic_parameter
        assert intrinsic_parameter[0, 2] == pytest.approx(50.0)
        assert intrinsic_parameter[1, 2] == pytest.approx(100.0)
        assert intrinsic_parameter[2, 2] == pytest.approx(1.0)

    def test_rejects_camera_fov_with_multiple_elements(self) -> None:
        """camera_fov with more than one element must raise ValueError."""
        with pytest.raises(ValueError, match="camera_fov must contain exactly one element"):
            FisheyeProcessorParameters(
                camera_fov=Angle(value=np.array([100.0, 200.0]), unit=AngleUnit.DEGREE),
            )

    @pytest.mark.parametrize("camera_fov_degree", [0.0, 360.0, 400.0])
    def test_rejects_camera_fov_outside_valid_range(self, camera_fov_degree: float) -> None:
        """camera_fov outside (0, 360) degrees must raise ValueError."""
        with pytest.raises(ValueError, match="camera_fov must satisfy"):
            FisheyeProcessorParameters(
                camera_fov=Angle(value=np.array([camera_fov_degree]), unit=AngleUnit.DEGREE),
            )

    @pytest.mark.parametrize("output_fov_degree", [0.0, 180.0, 270.0])
    def test_rejects_output_fov_outside_valid_range(self, output_fov_degree: float) -> None:
        """output_fov outside (0, 180) degrees must raise ValueError."""
        with pytest.raises(ValueError, match="output_fov must satisfy"):
            FisheyeProcessorParameters(
                output_fov=Angle(value=np.array([output_fov_degree]), unit=AngleUnit.DEGREE),
            )

    @pytest.mark.parametrize(("output_image_w", "output_image_h"), [(0, 960), (960, 0), (-1, 960)])
    def test_rejects_non_positive_output_image_size(self, output_image_w: int, output_image_h: int) -> None:
        """Non-positive output_image_w or output_image_h must raise ValueError."""
        with pytest.raises(ValueError, match="must be greater than zero"):
            FisheyeProcessorParameters(
                output_image_w=output_image_w,
                output_image_h=output_image_h,
            )
