"""The yaw/pitch/field of view captured from the interactive panorama viewer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapturedView:
    """
    Yaw/pitch/field of view captured from the panorama viewer's current camera.

    Sent back from the browser when the user clicks "Extract this view" in the
    panorama viewer's toolbar, after framing a shot by dragging and scrolling.

    Attributes
    ----------
    yaw_deg: float
        Longitude the camera was looking at when captured, in degrees (+: look right).
    pitch_deg: float
        Latitude the camera was looking at when captured, in degrees (+: look up).
    fov_deg: float
        The camera's field of view when captured, in degrees.
    """

    yaw_deg: float
    pitch_deg: float
    fov_deg: float
