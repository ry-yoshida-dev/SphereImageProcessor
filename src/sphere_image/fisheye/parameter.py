from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..utils import OutputFovBasis
from .method import FisheyeProjectionMethod

if TYPE_CHECKING:
    from .processor import FisheyeProcessor
from units import Angle, AngleUnit


@dataclass
class FisheyeProcessorParameters:
    """
    Parameters for fisheye processor(class for transforming fisheye image to equidistant image).

    Attributes
    ----------
    method: FisheyeProjectionMethod
        The method to use for transforming the fisheye image to equidistant image.
    is_camera_pointing_up: bool
        Whether the camera is pointing up or down (True: up, False: down).
    camera_fov: Angle
        Circular fisheye field of view of the camera.
    output_fov: Angle
        The base field of view of the output image.
    output_basis: OutputFovBasis
        Which axis `output_fov` is applied to.
    output_hfov: Angle
        The horizontal field of view of the output image.
    output_vfov: Angle
        The vertical field of view of the output image.
    output_image_w: int
        The width of the output image.
    output_image_h: int
        The height of the output image.

    """

    method: FisheyeProjectionMethod = field(
        default_factory=lambda: FisheyeProjectionMethod.EQUIDISTANT
    )
    is_camera_pointing_up: bool = True
    camera_fov: Angle = field(
        default_factory=lambda: Angle(value=np.array([185]), unit=AngleUnit.DEGREE)
    )

    output_image_w: int = 960
    output_image_h: int = 960

    output_fov: Angle = field(
        default_factory=lambda: Angle(value=np.array([90]), unit=AngleUnit.DEGREE)
    )
    output_basis: OutputFovBasis = field(
        default_factory=lambda: OutputFovBasis.VERTICAL
    )
    output_hfov: Angle = field(init=False)
    output_vfov: Angle = field(init=False)

    def __post_init__(self) -> None:
        self._validate_params()
        self.output_hfov, self.output_vfov = self.output_basis.build_output_fovs(
            output_fov=self.output_fov,
            aspect_ratio=self.aspect_ratio,
        )

    @property
    def aspect_ratio(self) -> float:
        """
        Output image aspect ratio (width / height).

        Returns
        -------
        float
            The aspect ratio.
        """
        return self.output_image_w / self.output_image_h

    def build_processor(self, image: np.ndarray) -> FisheyeProcessor:
        """
        Build a fisheye processor configured with this parameter set.

        Parameters
        ----------
        image: np.ndarray
            The image to process.

        Returns
        -------
        FisheyeProcessor
            Processor instance that applies self.method in radius mapping.
        """
        from .processor import FisheyeProcessor

        return FisheyeProcessor(image=image, params=self)

    @property
    def intrinsic_parameter(self) -> np.ndarray:
        """
        Build a pinhole-style intrinsic matrix from output image size and FoV.

        Notes
        -----
        The principal point is assumed at the image center and focal lengths are
        derived from horizontal/vertical FoV respectively.

        Returns
        -------
        np.ndarray
            The intrinsic parameter matrix.
        """
        image_width = self.output_image_w
        image_height = self.output_image_h

        principal_x = image_width / 2.0
        principal_y = image_height / 2.0

        focal_x = image_width / (2.0 * np.tan(self.output_hfov.radian[0] / 2.0))
        focal_y = image_height / (2.0 * np.tan(self.output_vfov.radian[0] / 2.0))

        return np.array(
            [
                [focal_x, 0.0, principal_x],
                [0.0, focal_y, principal_y],
                [0, 0, 1],
            ]
        )

    def _validate_params(self) -> None:
        """
        Validate basic shape/value constraints for constructor inputs.

        Raises
        ------
        ValueError
            If the parameters are invalid.
            - camera_fov and output_fov must contain exactly one element.
            - camera_fov must satisfy 0 < fov < 360 degrees.
            - output_fov must satisfy 0 < fov < 180 degrees.
            - output_fov must contain exactly one element.
            - output_image_w and output_image_h must be greater than zero.
        """
        # Current implementation expects scalar angles encoded as length-1 arrays.
        for field_name, angle in (
            ("camera_fov", self.camera_fov),
            ("output_fov", self.output_fov),
        ):
            value = np.asarray(angle.value)
            if value.size != 1:
                raise ValueError(
                    f"{field_name} must contain exactly one element, got {value.size}."
                )
        camera_fov_radian = float(
            np.asarray(self.camera_fov.radian, dtype=np.float64).reshape(-1)[0]
        )
        if not (0.0 < camera_fov_radian < 2.0 * np.pi):
            raise ValueError(
                f"camera_fov must satisfy 0 < fov < 360 degrees, got {np.rad2deg(camera_fov_radian):.3f} degrees."
            )

        output_fov_radian = float(
            np.asarray(self.output_fov.radian, dtype=np.float64).reshape(-1)[0]
        )
        if not (0.0 < output_fov_radian < np.pi):
            raise ValueError(
                f"output_fov must satisfy 0 < fov < 180 degrees, got {np.rad2deg(output_fov_radian):.3f} degrees."
            )
        if self.output_image_h <= 0 or self.output_image_w <= 0:
            raise ValueError(
                f"output_image_w and output_image_h must be greater than zero, got w={self.output_image_w} and h={self.output_image_h}."
            )
