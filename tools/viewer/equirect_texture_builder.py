from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from units import Angle, AngleUnit

from sphere_image.fisheye import FisheyeProjectionMethod


@dataclass(frozen=True)
class FisheyeEquirectTextureBuilder:
    """
    Converts a circular fisheye image into a full equirectangular texture.

    Every texture pixel is treated as the center ray of an imaginary
    perspective view aimed at that pixel's longitude/latitude, so the
    incident-angle math is identical to `FisheyeProcessor`'s: the same
    `FisheyeProjectionMethod.calculate_radius` equation is inverted directly,
    with no intermediate perspective renders.

    Attributes
    ----------
    method: FisheyeProjectionMethod
        Fisheye radius equation to invert.
    camera_fov_degree: float
        Circular fisheye field of view, in degrees.
    is_camera_pointing_up: bool
        Whether the camera is mounted pointing upward. `FisheyeProcessor`'s
        own incident-angle formula always treats the disk's center as the
        world zenith, regardless of this flag; a downward-pointing camera is
        instead handled by flipping the finished texture vertically, so its
        captured content ends up at the nadir (bottom) instead of the
        zenith (top).
    texture_width: int
        Output texture width, in pixels.
    texture_height: int
        Output texture height, in pixels.
    """

    method: FisheyeProjectionMethod
    camera_fov_degree: float
    is_camera_pointing_up: bool
    texture_width: int
    texture_height: int

    def build(self, image: np.ndarray) -> np.ndarray:
        """
        Build the equirectangular texture from a source fisheye image.

        Parameters
        ----------
        image: np.ndarray
            Source circular fisheye image.

        Returns
        -------
        np.ndarray
            Equirectangular texture, shape (texture_height, texture_width, 3).
            Longitude 0 is at the texture's horizontal center; latitude
            +90 degrees (zenith) is the top row, -90 degrees (nadir) is the
            bottom row. Pixels outside the fisheye's captured cone are black.
        """
        longitude_grid, latitude_grid = self._build_longitude_latitude_grid()

        incident_angle_grid = np.pi / 2 - latitude_grid
        half_camera_fov_radian = np.deg2rad(self.camera_fov_degree) / 2.0
        incident_angle = Angle(value=incident_angle_grid.reshape(-1), unit=AngleUnit.RADIAN)
        radius = self.method.calculate_radius(f=half_camera_fov_radian, angle=incident_angle)
        radius = radius.reshape(incident_angle_grid.shape)

        u_coordinate = radius * np.cos(longitude_grid)
        v_coordinate = radius * np.sin(longitude_grid)
        normalized_u = 0.5 + 0.5 * u_coordinate
        normalized_v = 0.5 + 0.5 * v_coordinate

        x_pixel_coordinate = (normalized_u * (image.shape[1] - 1)).astype(np.float32)
        y_pixel_coordinate = (normalized_v * (image.shape[0] - 1)).astype(np.float32)
        texture = cv2.remap(
            image,
            x_pixel_coordinate,
            y_pixel_coordinate,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if self.is_camera_pointing_up:
            return texture
        return cv2.flip(texture, 0)

    def _build_longitude_latitude_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the per-texel longitude/latitude grid, both in radians.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Longitude and latitude grids, each shape
            (texture_height, texture_width).
        """
        longitude = np.linspace(-np.pi, np.pi, self.texture_width, endpoint=False)
        latitude = np.linspace(np.pi / 2, -np.pi / 2, self.texture_height)
        return np.meshgrid(longitude, latitude)
