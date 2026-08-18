from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from sphere_image.fisheye import FisheyeProjectionMethod

from .equirect_texture_builder import FisheyeEquirectTextureBuilder
from .image_kind import ImageKind


@dataclass(frozen=True)
class ProjectionSettings:
    """
    User-configurable settings for building the panorama viewer's texture.

    Attributes
    ----------
    image_kind: ImageKind
        Selected source image projection family.
    fisheye_method: FisheyeProjectionMethod
        Radius equation used when `image_kind` is `FISHEYE`.
    camera_fov_degree: float
        Circular fisheye field of view, in degrees. Only used for `FISHEYE`.
    is_camera_pointing_up: bool
        Whether the camera is mounted pointing upward. Only used for
        `FISHEYE`.
    texture_width: int
        Equirectangular texture width, in pixels.
    texture_height: int
        Equirectangular texture height, in pixels.
    """

    image_kind: ImageKind
    fisheye_method: FisheyeProjectionMethod
    camera_fov_degree: float
    is_camera_pointing_up: bool
    texture_width: int
    texture_height: int

    def build_equirect_texture(self, image: np.ndarray) -> np.ndarray:
        """
        Build the equirectangular texture matching `image_kind` for `image`.

        Parameters
        ----------
        image: np.ndarray
            Source image to convert.

        Returns
        -------
        np.ndarray
            Equirectangular texture, shape (texture_height, texture_width, 3).
        """
        match self.image_kind:
            case ImageKind.FISHEYE:
                builder = FisheyeEquirectTextureBuilder(
                    method=self.fisheye_method,
                    camera_fov_degree=self.camera_fov_degree,
                    is_camera_pointing_up=self.is_camera_pointing_up,
                    texture_width=self.texture_width,
                    texture_height=self.texture_height,
                )
                return builder.build(image)
            case ImageKind.EQUIRECTANGULAR:
                return cv2.resize(
                    image,
                    (self.texture_width, self.texture_height),
                    interpolation=cv2.INTER_AREA,
                )
