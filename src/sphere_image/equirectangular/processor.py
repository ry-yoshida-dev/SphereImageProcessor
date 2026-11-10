from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from rotation import RotationMatrix

from ..utils import SphericalCoordinates
from .method import EquirectangularProjectionMethod
from .parameter import EquirectangularProcessorParameters


@dataclass
class EquirectangularProcessor:
    """
    Equirectangular image remapping processor.

    Attributes
    ----------
    image: np.ndarray
        Source equirectangular image.
    params: EquirectangularProcessorParameters
        Parameters controlling projection and output geometry.
    """

    image: np.ndarray
    params: EquirectangularProcessorParameters = field(
        default_factory=EquirectangularProcessorParameters
    )

    def _map_rotation_to_uv(
        self,
        rotation_matrix: RotationMatrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return normalized source image coordinates (u, v) for remap.

        Parameters
        ----------
        rotation_matrix: RotationMatrix
            Rotation matrix.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The u and v coordinates.

        Raises
        ------
        ValueError
            If the selected method is not implemented.
        """
        if self.params.method is not EquirectangularProjectionMethod.PERSPECTIVE:
            raise ValueError(f"Not implemented error. -> method: {self.params.method}")

        direction_vectors = self._create_direction_vector_grid()
        rotated_direction_vectors = direction_vectors @ rotation_matrix.T

        x_coordinates = rotated_direction_vectors[:, 0]
        y_coordinates = rotated_direction_vectors[:, 1]
        z_coordinates = rotated_direction_vectors[:, 2]
        spherical_coordinates = SphericalCoordinates.from_cartesian(
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            z_coordinates=z_coordinates,
        )
        u_coordinates = spherical_coordinates.u_coordinates
        v_coordinates = spherical_coordinates.v_coordinates

        return (
            u_coordinates.reshape(
                self.params.output_image_h, self.params.output_image_w
            ),
            v_coordinates.reshape(
                self.params.output_image_h, self.params.output_image_w
            ),
        )

    def run_pipeline(self, rotation_matrix: RotationMatrix) -> np.ndarray:
        """
        Run the equirectangular remapping pipeline.

        Parameters
        ----------
        rotation_matrix: RotationMatrix
            Rotation matrix applied before projecting to UV.

        Returns
        -------
        np.ndarray
            The remapped image.
        """
        u_coordinates, v_coordinates = self._map_rotation_to_uv(
            rotation_matrix=rotation_matrix
        )
        return self.remap(
            u_coordinates=u_coordinates,
            v_coordinates=v_coordinates,
        )

    def _create_direction_vector_grid(self) -> np.ndarray:
        """
        Generate normalized 3D direction vectors for each output pixel.

        Returns
        -------
        np.ndarray
            Direction vectors shaped as (H*W, 3).
        """
        horizontal_coordinates = np.linspace(
            -np.tan(self.params.output_hfov.radian / 2),
            np.tan(self.params.output_hfov.radian / 2),
            self.params.output_image_w,
        )
        vertical_coordinates = np.linspace(
            -np.tan(self.params.output_vfov.radian / 2),
            np.tan(self.params.output_vfov.radian / 2),
            self.params.output_image_h,
        )

        if self.params.is_camera_pointing_up:
            vertical_coordinates *= -1

        horizontal_grid, vertical_grid = np.meshgrid(
            horizontal_coordinates,
            vertical_coordinates,
        )
        depth_grid = np.ones_like(horizontal_grid)
        direction_vectors = np.stack(
            [depth_grid, horizontal_grid, vertical_grid],
            axis=-1,
        )
        direction_vectors /= np.linalg.norm(direction_vectors, axis=-1, keepdims=True)
        return direction_vectors.reshape(-1, 3)

    def remap(
        self,
        u_coordinates: np.ndarray,
        v_coordinates: np.ndarray,
    ) -> np.ndarray:
        """
        Remap the image using normalized source coordinates.

        Parameters
        ----------
        u_coordinates: np.ndarray
            Normalized horizontal coordinates in source image space.
        v_coordinates: np.ndarray
            Normalized vertical coordinates in source image space.

        Returns
        -------
        np.ndarray
            The remapped image.
        """
        x_pixel_coordinate = (u_coordinates * (self.image.shape[1] - 1)).astype(
            np.float32
        )
        y_pixel_coordinate = (v_coordinates * (self.image.shape[0] - 1)).astype(
            np.float32
        )
        return cv2.remap(
            self.image,
            x_pixel_coordinate,
            y_pixel_coordinate,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

    @classmethod
    def from_path(
        cls,
        path: str,
        params: EquirectangularProcessorParameters | None = None,
    ) -> EquirectangularProcessor:
        """
        Build a processor by reading an image from disk.

        Parameters
        ----------
        path: str
            Input image path.
        params: EquirectangularProcessorParameters | None
            Optional parameter set. Defaults to freshly constructed parameters.

        Returns
        -------
        EquirectangularProcessor
            Processor initialized with loaded image and selected parameters.

        Raises
        ------
        ValueError
            If the image cannot be read from the provided path.
        """
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to read image: {path}")

        selected_params = params or EquirectangularProcessorParameters()
        return cls(image=image, params=selected_params)
