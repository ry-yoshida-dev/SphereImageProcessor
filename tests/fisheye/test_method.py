from __future__ import annotations

import numpy as np
import pytest
from units import Angle, AngleUnit

from sphere_image.fisheye import FisheyeProjectionMethod


class TestFisheyeProjectionMethodCalculateRadius:
    """Tests for FisheyeProjectionMethod.calculate_radius."""

    @pytest.mark.parametrize(
        "method",
        [
            FisheyeProjectionMethod.EQUIDISTANT,
            FisheyeProjectionMethod.ORTHOGRAPHIC,
            FisheyeProjectionMethod.STEREOGRAPHIC,
            FisheyeProjectionMethod.EQUISOLID,
        ],
    )
    def test_radius_is_zero_at_optical_axis(self, method: FisheyeProjectionMethod) -> None:
        """Every projection law must map theta=0 to radius=0."""
        half_fov_radian = np.pi / 2
        angle = Angle(value=np.array([0.0]), unit=AngleUnit.RADIAN)
        radius = method.calculate_radius(f=half_fov_radian, angle=angle)
        assert radius[0] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "method",
        [
            FisheyeProjectionMethod.EQUIDISTANT,
            FisheyeProjectionMethod.ORTHOGRAPHIC,
            FisheyeProjectionMethod.STEREOGRAPHIC,
            FisheyeProjectionMethod.EQUISOLID,
        ],
    )
    def test_radius_is_one_at_rim(self, method: FisheyeProjectionMethod) -> None:
        """Every projection law must map theta=f to radius=1 at the fisheye rim."""
        half_fov_radian = np.pi / 2
        angle = Angle(value=np.array([half_fov_radian]), unit=AngleUnit.RADIAN)
        radius = method.calculate_radius(f=half_fov_radian, angle=angle)
        assert radius[0] == pytest.approx(1.0)

    def test_equidistant_radius_is_linear_in_angle(self) -> None:
        """The equidistant law must produce a radius directly proportional to theta."""
        half_fov_radian = np.pi / 2
        angle = Angle(value=np.array([half_fov_radian / 2]), unit=AngleUnit.RADIAN)
        radius = FisheyeProjectionMethod.EQUIDISTANT.calculate_radius(
            f=half_fov_radian, angle=angle
        )
        assert radius[0] == pytest.approx(0.5)
