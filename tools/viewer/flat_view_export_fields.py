"""Manually entered fields for a flat_views.yaml export, excluding yaw/pitch/FoV."""

from __future__ import annotations

from dataclasses import dataclass

from sphere_image.utils import OutputFovBasis


@dataclass(frozen=True)
class FlatViewExportFields:
    """
    Sidebar-entered flat_views.yaml fields that are not captured from the panorama viewer.

    Yaw, pitch, and field of view instead come from a `CapturedView`, captured by the
    user from the panorama viewer's own toolbar.

    Attributes
    ----------
    camera_name: str
        Name of the physical camera this view is rendered from (e.g. "camera3").
    view_name: str
        Name under which this view is registered in `flat_views.yaml`.
    roll_deg: float
        Roll angle in degrees (+: clockwise image tilt).
    output_width: int
        Width of the rendered view, in pixels.
    output_height: int
        Height of the rendered view, in pixels.
    output_basis: OutputFovBasis
        Which axis the exported field of view is applied to.
    """

    camera_name: str
    view_name: str
    roll_deg: float
    output_width: int
    output_height: int
    output_basis: OutputFovBasis
