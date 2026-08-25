"""Settings for exporting the panorama viewer's current view as a flat_views.yaml entry."""

from __future__ import annotations

from dataclasses import dataclass

from sphere_image.utils import OutputFovBasis


@dataclass(frozen=True)
class FlatViewExportSettings:
    """
    Sphere2flat flat view parameters for the view currently shown in the panorama viewer.

    These mirror the ``camera``/``yaw_deg``/``pitch_deg``/``roll_deg``/``output_width``/
    ``output_height``/``output_fov_deg``/``output_basis`` keys of one named entry in a
    ``flat_views.yaml`` file, as consumed by ``SphereImageCalibration``.

    Attributes
    ----------
    camera_name: str
        Name of the physical camera this view is rendered from (e.g. "camera3").
    view_name: str
        Name under which this view is registered in ``flat_views.yaml``.
    yaw_deg: float
        Yaw angle in degrees (+: look right).
    pitch_deg: float
        Pitch angle in degrees (+: look up).
    roll_deg: float
        Roll angle in degrees (+: clockwise image tilt).
    output_width: int
        Width of the rendered view, in pixels.
    output_height: int
        Height of the rendered view, in pixels.
    output_fov_deg: float
        Field of view applied along the `output_basis` axis, in degrees.
    output_basis: OutputFovBasis
        Which axis `output_fov_deg` is applied to.
    """

    camera_name: str
    view_name: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    output_width: int
    output_height: int
    output_fov_deg: float
    output_basis: OutputFovBasis

    def to_flat_views_yaml_snippet(self) -> str:
        """
        Render this view as one named entry, formatted like ``flat_views.yaml``.

        Returns
        -------
        str
            A YAML snippet whose only top-level key is `view_name` (or the
            placeholder "view_name" when it is blank).
        """
        entry_name = self.view_name or "view_name"
        camera_name = self.camera_name or "camera_name"
        lines = [
            f"{entry_name}:",
            f"  camera: {camera_name}",
            f"  yaw_deg: {self.yaw_deg}",
            f"  pitch_deg: {self.pitch_deg}",
            f"  roll_deg: {self.roll_deg}",
            f"  output_width: {self.output_width}",
            f"  output_height: {self.output_height}",
            f"  output_fov_deg: {self.output_fov_deg}",
            f"  output_basis: {self.output_basis.value}",
        ]
        return "\n".join(lines) + "\n"
