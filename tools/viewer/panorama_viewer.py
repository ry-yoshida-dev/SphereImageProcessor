from __future__ import annotations

import base64
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import streamlit.components.v1 as components

from sphere_image.fisheye import FisheyeProjectionMethod

from .captured_view import CapturedView
from .image_kind import ImageKind

_FRONTEND_DIRECTORY = Path(__file__).parent / "frontend"

_component_func = components.declare_component("panorama_viewer", path=str(_FRONTEND_DIRECTORY))


class PanoramaViewer:
    """
    Client-side WebGL panorama viewer, embedded as a bidirectional Streamlit component.

    The equirectangular texture is generated once in Python and handed to the
    browser; looking around (drag) and zooming (scroll) then run entirely in
    JavaScript at the display's own frame rate, with no Streamlit rerun. Clicking
    "Extract this view" in its toolbar sends the camera's current yaw, pitch,
    and field of view back to Python.
    """

    @staticmethod
    def render(
        texture_bgr: np.ndarray,
        roll_degree: float,
        aspect_ratio: float,
        max_width_px: int,
        image_kind: ImageKind,
        fisheye_method: FisheyeProjectionMethod,
        camera_fov_degree: float,
        is_camera_pointing_up: bool,
        source_image_width: int,
        source_image_height: int,
        key: str,
    ) -> CapturedView | None:
        """
        Encode `texture_bgr` and render the panorama viewer component.

        Parameters
        ----------
        texture_bgr: np.ndarray
            Equirectangular texture, in BGR channel order.
        roll_degree: float
            Fixed clockwise tilt of the camera's up vector, in degrees. Unlike yaw
            and pitch, this is not adjustable by dragging, so it is applied live on
            every render.
        aspect_ratio: float
            Width divided by height of the rendered viewer canvas. The
            canvas fills the available width (up to `max_width_px`) and
            derives its height from this ratio (via CSS `aspect-ratio`), so
            it stays well-proportioned regardless of the browser's window
            width.
        max_width_px: int
            Maximum width of the rendered viewer canvas, in pixels. Caps how
            large the canvas grows on wide windows/columns.
        image_kind: ImageKind
            Source image projection family, used to invert the cursor's
            sphere position back into source-image coordinates.
        fisheye_method: FisheyeProjectionMethod
            Fisheye radius equation to invert. Only used when `image_kind`
            is `FISHEYE`.
        camera_fov_degree: float
            Circular fisheye field of view, in degrees. Only used when
            `image_kind` is `FISHEYE`.
        is_camera_pointing_up: bool
            Whether the camera is mounted pointing upward. Only used when
            `image_kind` is `FISHEYE`.
        source_image_width: int
            Width, in pixels, of the original source image.
        source_image_height: int
            Height, in pixels, of the original source image.
        key: str
            Stable Streamlit widget key. Must stay constant across reruns so a
            captured view survives unrelated sidebar changes (e.g. switching
            images), which would otherwise change this component's identity.

        Returns
        -------
        CapturedView | None
            The most recently captured yaw/pitch/field of view, or `None` if
            "Extract this view" has not been clicked yet.

        Raises
        ------
        ValueError
            If `texture_bgr` cannot be encoded as JPEG.
        """
        is_encoded, encoded_texture = cv2.imencode(
            ".jpg", texture_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not is_encoded:
            raise ValueError("Failed to encode the panorama texture as JPEG.")
        texture_base64 = base64.b64encode(encoded_texture.tobytes()).decode("ascii")

        raw_captured_view = cast(
            "dict[str, float] | None",
            _component_func(
                texture_base64=texture_base64,
                roll_degree=roll_degree,
                aspect_ratio=aspect_ratio,
                max_width_px=max_width_px,
                image_kind=image_kind.value,
                fisheye_method=fisheye_method.value,
                camera_fov_degree=camera_fov_degree,
                is_camera_pointing_up=is_camera_pointing_up,
                source_image_width=source_image_width,
                source_image_height=source_image_height,
                default=None,
                key=key,
            ),
        )
        if raw_captured_view is None:
            return None
        return CapturedView(
            yaw_deg=float(raw_captured_view["yaw_deg"]),
            pitch_deg=float(raw_captured_view["pitch_deg"]),
            fov_deg=float(raw_captured_view["fov_deg"]),
        )
