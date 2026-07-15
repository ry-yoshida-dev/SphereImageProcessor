from __future__ import annotations

from sphere_image.equirectangular import EquirectangularProjectionMethod


class TestEquirectangularProjectionMethod:
    """Tests for the EquirectangularProjectionMethod enum."""

    def test_members_expose_expected_display_names(self) -> None:
        """Each member's value must be its human-readable projection name."""
        expected_values_by_member = {
            EquirectangularProjectionMethod.PERSPECTIVE: "Perspective",
            EquirectangularProjectionMethod.ORTHOGRAPHIC: "Orthographic",
            EquirectangularProjectionMethod.STEREOGRAPHIC: "Stereographic",
            EquirectangularProjectionMethod.EQUIDISTANT: "Equidistant",
            EquirectangularProjectionMethod.EQUAL_AREA: "EqualArea",
            EquirectangularProjectionMethod.CYLINDRICAL: "Cylindrical",
            EquirectangularProjectionMethod.MERCATOR: "Mercator",
            EquirectangularProjectionMethod.LAMBERT_CYLINDRICAL: "LambertCylindrical",
        }
        for member, expected_value in expected_values_by_member.items():
            assert member.value == expected_value
