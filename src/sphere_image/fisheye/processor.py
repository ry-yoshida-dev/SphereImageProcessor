from dataclasses import dataclass, field

import cv2
import numpy as np
from geometry.planar import PolarCoordinate
from geometry.spatial import Vectors3D
from rotation import RotationMatrix

from .parameter import FisheyeProcessorParameters


@dataclass
class FisheyeProcessor:
    """
    Fisheye processor.

    Attributes
    ----------
    image: np.ndarray
        The image to process.
    params: FisheyeProcessorParameters
        The parameters for the fisheye processor.
    """

    image: np.ndarray
    params: FisheyeProcessorParameters = field(
        default_factory=FisheyeProcessorParameters
    )

    def run_pipeline(self, rotation_matrix: RotationMatrix) -> np.ndarray:
        """
        Run the fisheye remapping pipeline.

        Parameters
        ----------
        rotation_matrix: RotationMatrix
            Rotation matrix.

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

    def _map_rotation_to_uv(
        self,
        rotation_matrix: RotationMatrix,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return normalized image-plane (u, v) in [0, 1] for remap.

        Parameters
        ----------
        rotation_matrix: RotationMatrix
            Rotation matrix.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The u and v coordinates.
        """
        direction_vectors: Vectors3D = self._create_direction_vector_grid()
        rotated_direction_vectors: Vectors3D = Vectors3D(
            value=direction_vectors.value @ rotation_matrix.T
        )
        azimuthal_angles = rotated_direction_vectors.to_azimuthal_angles(up_index=2)
        polar_angles = rotated_direction_vectors.to_polar_angles(
            forward_index=0,
            right_index=1,
        )
        max_incident_angle = float(self.params.camera_fov.radian[0] / 2)
        radius = self.params.method.calculate_radius(
            f=max_incident_angle,
            angle=azimuthal_angles,
        )
        polar_coordinate = PolarCoordinate(
            radius=radius,
            angle=polar_angles,
        )
        return self._PolarCoordinate2NormalizedCartesianCoordinate(
            polar_coordinate=polar_coordinate
        )

    def _create_direction_vector_grid(self) -> Vectors3D:
        """
        Generate a 3D direction vector for each pixel in the output image.

        The function constructs a grid of normalized 3D vectors that represent
        the viewing direction corresponding to each pixel in the rectified
        output image. The grid is defined based on the specified horizontal
        and vertical field of view (FOV).

        Returns:
        --------
        Vectors3D
            A (H, W, 3) array of normalized direction vectors on camera coordinate system,
            where H and W correspond to the output image height and width.
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
            horizontal_coordinates, vertical_coordinates
        )
        depth_grid = np.ones_like(horizontal_grid)

        direction_vectors = np.stack(
            [depth_grid, horizontal_grid, vertical_grid], axis=-1
        )
        direction_vectors /= np.linalg.norm(direction_vectors, axis=-1, keepdims=True)
        return Vectors3D(value=direction_vectors.reshape(-1, 3))

    def _PolarCoordinate2NormalizedCartesianCoordinate(
        self,
        polar_coordinate: PolarCoordinate,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert polar coordinate to normalized cartesian coordinate.
        Polar coordinates: radius ∈ [0, ∞), angle ∈ [0, 2π) or [-π, π]

        Parameters
        ----------
        polar_coordinate: PolarCoordinate
            The polar coordinate.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The u and v coordinates.
        """
        u_coordinates: np.ndarray = polar_coordinate.u
        v_coordinates: np.ndarray = polar_coordinate.v
        normed_u_coordinates: np.ndarray = 0.5 + 0.5 * (u_coordinates)
        normed_v_coordinates: np.ndarray = 0.5 + 0.5 * (v_coordinates)
        return (
            normed_u_coordinates.reshape(
                self.params.output_image_h, self.params.output_image_w
            ),
            normed_v_coordinates.reshape(
                self.params.output_image_h, self.params.output_image_w
            ),
        )

    def remap(
        self,
        u_coordinates: np.ndarray,
        v_coordinates: np.ndarray,
    ) -> np.ndarray:
        """
        Remap the image using the u and v coordinates.

        Parameters
        ----------
        u_coordinates: np.ndarray
            The u coordinates (horizontal axis of image(screen) coordinate system).
        v_coordinates: np.ndarray
            The v coordinates (vertical axis of image(screen) coordinate system).

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

        remapped_image = cv2.remap(
            self.image,
            x_pixel_coordinate,
            y_pixel_coordinate,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return remapped_image
