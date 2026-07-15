from __future__ import annotations

import numpy as np
import pytest
from units import Angle, AngleUnit

from sphere_image.utils import OutputFovBasis


class TestOutputFovBasis:
    """Tests for OutputFovBasis.build_output_fovs."""

    def test_vertical_basis_preserves_input_fov(self) -> None:
        """The vertical basis must return the input angle unchanged as output_vfov."""
        output_fov = Angle(value=np.array([90.0]), unit=AngleUnit.DEGREE)
        output_hfov, output_vfov = OutputFovBasis.VERTICAL.build_output_fovs(
            output_fov=output_fov,
            aspect_ratio=16.0 / 9.0,
        )
        assert output_vfov.degree == pytest.approx(90.0)
        assert output_hfov.degree > output_vfov.degree

    def test_horizontal_basis_preserves_input_fov(self) -> None:
        """The horizontal basis must return the input angle unchanged as output_hfov."""
        output_fov = Angle(value=np.array([90.0]), unit=AngleUnit.DEGREE)
        output_hfov, output_vfov = OutputFovBasis.HORIZONTAL.build_output_fovs(
            output_fov=output_fov,
            aspect_ratio=16.0 / 9.0,
        )
        assert output_hfov.degree == pytest.approx(90.0)
        assert output_vfov.degree < output_hfov.degree

    def test_square_aspect_ratio_yields_equal_fovs(self) -> None:
        """With a square aspect ratio, horizontal and vertical FoV must match."""
        output_fov = Angle(value=np.array([60.0]), unit=AngleUnit.DEGREE)
        output_hfov, output_vfov = OutputFovBasis.VERTICAL.build_output_fovs(
            output_fov=output_fov,
            aspect_ratio=1.0,
        )
        assert output_hfov.degree == pytest.approx(output_vfov.degree)
        assert output_hfov.degree == pytest.approx(60.0)

    def test_pinhole_tangent_relationship_holds(self) -> None:
        """tan(half_hfov) / tan(half_vfov) must equal the aspect ratio."""
        aspect_ratio = 4.0 / 3.0
        output_fov = Angle(value=np.array([75.0]), unit=AngleUnit.DEGREE)
        output_hfov, output_vfov = OutputFovBasis.VERTICAL.build_output_fovs(
            output_fov=output_fov,
            aspect_ratio=aspect_ratio,
        )
        ratio = np.tan(output_hfov.radian / 2.0) / np.tan(output_vfov.radian / 2.0)
        assert ratio.item() == pytest.approx(aspect_ratio)
