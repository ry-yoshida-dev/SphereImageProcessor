from __future__ import annotations

import numpy as np
import pytest

from sphere_image.utils import SphericalCoordinates


class TestSphericalCoordinatesHypotenuse:
    """Tests for SphericalCoordinates.hypotenuse."""

    def test_computes_euclidean_norm(self) -> None:
        """hypotenuse must equal sqrt(x^2 + y^2) elementwise."""
        x_coordinates = np.array([3.0, 0.0, 1.0])
        y_coordinates = np.array([4.0, 5.0, 1.0])
        result = SphericalCoordinates.hypotenuse(
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
        )
        np.testing.assert_allclose(result, [5.0, 5.0, np.sqrt(2.0)])


class TestSphericalCoordinatesFromCartesian:
    """Tests for SphericalCoordinates.from_cartesian."""

    def test_forward_axis_maps_to_zero_longitude_and_latitude(self) -> None:
        """A unit vector along +x must map to longitude=0 and latitude=0."""
        coordinates = SphericalCoordinates.from_cartesian(
            x_coordinates=np.array([1.0]),
            y_coordinates=np.array([0.0]),
            z_coordinates=np.array([0.0]),
        )
        assert coordinates.longitude[0] == pytest.approx(0.0)
        assert coordinates.latitude[0] == pytest.approx(0.0)

    def test_right_axis_maps_to_quarter_turn_longitude(self) -> None:
        """A unit vector along +y must map to longitude=+pi/2."""
        coordinates = SphericalCoordinates.from_cartesian(
            x_coordinates=np.array([0.0]),
            y_coordinates=np.array([1.0]),
            z_coordinates=np.array([0.0]),
        )
        assert coordinates.longitude[0] == pytest.approx(np.pi / 2)

    def test_up_axis_maps_to_north_pole_latitude(self) -> None:
        """A unit vector along +z must map to latitude=+pi/2."""
        coordinates = SphericalCoordinates.from_cartesian(
            x_coordinates=np.array([0.0]),
            y_coordinates=np.array([0.0]),
            z_coordinates=np.array([1.0]),
        )
        assert coordinates.latitude[0] == pytest.approx(np.pi / 2)


class TestSphericalCoordinatesUvCoordinates:
    """Tests for SphericalCoordinates.u_coordinates and v_coordinates."""

    def test_zero_longitude_maps_to_center_u(self) -> None:
        """longitude=0 must map to u=0.5."""
        coordinates = SphericalCoordinates(
            longitude=np.array([0.0]),
            latitude=np.array([0.0]),
        )
        assert coordinates.u_coordinates[0] == pytest.approx(0.5)

    def test_negative_quarter_turn_longitude_maps_to_quarter_u(self) -> None:
        """longitude=-pi/2 must map to u=0.25."""
        coordinates = SphericalCoordinates(
            longitude=np.array([-np.pi / 2]),
            latitude=np.array([0.0]),
        )
        assert coordinates.u_coordinates[0] == pytest.approx(0.25)

    def test_zero_latitude_maps_to_center_v(self) -> None:
        """latitude=0 must map to v=0.5."""
        coordinates = SphericalCoordinates(
            longitude=np.array([0.0]),
            latitude=np.array([0.0]),
        )
        assert coordinates.v_coordinates[0] == pytest.approx(0.5)

    def test_north_pole_latitude_is_clamped_to_zero(self) -> None:
        """latitude=+pi/2 (north pole) must map to v=0."""
        coordinates = SphericalCoordinates(
            longitude=np.array([0.0]),
            latitude=np.array([np.pi / 2]),
        )
        assert coordinates.v_coordinates[0] == pytest.approx(0.0)

    def test_south_pole_latitude_is_clamped_to_one(self) -> None:
        """latitude=-pi/2 (south pole) must map to v=1."""
        coordinates = SphericalCoordinates(
            longitude=np.array([0.0]),
            latitude=np.array([-np.pi / 2]),
        )
        assert coordinates.v_coordinates[0] == pytest.approx(1.0)
