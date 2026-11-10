from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from units import Angle, AngleUnit

from ..utils import OutputFovBasis as EquirectangularBasis
from .method import EquirectangularProjectionMethod

if TYPE_CHECKING:
    from .processor import EquirectangularProcessor


@dataclass
class EquirectangularProcessorParameters:
    """
    Parameters for equirectangular processor.

    Attributes
    ----------
    method: EquirectangularProjectionMethod
        Projection method used by the processor.
    is_camera_pointing_up: bool
        Whether the camera is mounted upward (True) or downward (False).
    output_fov: Angle
        Base field of view for the output image.
    output_basis: EquirectangularBasis
        Axis basis that `output_fov` is applied to.
    output_image_w: int
        Width of the output image in pixels.
    output_image_h: int
        Height of the output image in pixels.
    output_hfov: Angle
        Derived horizontal output field of view.
    output_vfov: Angle
        Derived vertical output field of view.
    """

    method: EquirectangularProjectionMethod = field(
        default_factory=lambda: EquirectangularProjectionMethod.PERSPECTIVE
    )
    is_camera_pointing_up: bool = True
    output_fov: Angle = field(
        default_factory=lambda: Angle(value=np.array([90]), unit=AngleUnit.DEGREE)
    )
    output_basis: EquirectangularBasis = field(
        default_factory=lambda: EquirectangularBasis.VERTICAL
    )
    output_image_w: int = 1280
    output_image_h: int = 720
    output_hfov: Angle = field(init=False)
    output_vfov: Angle = field(init=False)

    def __post_init__(self) -> None:
        self.validate_params()
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

    def build_processor(self, image: np.ndarray) -> EquirectangularProcessor:
        """
        Build an equirectangular processor configured with this parameter set.

        Parameters
        ----------
        image: np.ndarray
            The image to process.

        Returns
        -------
        EquirectangularProcessor
            Processor instance configured with this parameter set.
        """
        from .processor import EquirectangularProcessor

        return EquirectangularProcessor(image=image, params=self)

    def validate_params(self) -> None:
        """
        Validate basic shape/value constraints for constructor inputs.

        Raises
        ------
        ValueError
            If the parameters are invalid.
            - output_fov must contain exactly one element.
            - output_image_w and output_image_h must be greater than zero.
        """
        value = np.asarray(self.output_fov.value)
        if value.size != 1:
            raise ValueError(
                f"output_fov must contain exactly one element, got {value.size}."
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
