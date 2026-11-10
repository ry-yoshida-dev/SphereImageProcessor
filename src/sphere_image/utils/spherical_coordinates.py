from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SphericalCoordinates:
    """
    Spherical coordinates represented by longitude and latitude in radians.

    longitude and latitude are each np.ndarray with shape (n,) or (h, w).
    Both arrays share the same shape and represent per-sample angular positions
    on the unit sphere.
    """

    longitude: np.ndarray
    latitude: np.ndarray

    @property
    def u_coordinates(self) -> np.ndarray:
        """
        Convert longitude to normalized equirectangular u coordinates.

        Returns
        -------
        np.ndarray
            shape (n,) or (h, w)
            Horizontal texture coordinate in [0, 1), where 0.5 corresponds to
            longitude 0.
        """
        return np.mod((self.longitude / (2 * np.pi)) + 0.5, 1.0)

    @property
    def v_coordinates(self) -> np.ndarray:
        """
        Convert latitude to normalized equirectangular v coordinates.

        Returns
        -------
        np.ndarray
            shape (n,) or (h, w)
            Vertical texture coordinate in [0, 1], where 0.5 corresponds to
            latitude 0 and poles are clamped to the range edges.
        """
        return np.clip(0.5 - (self.latitude / np.pi), 0.0, 1.0)

    @classmethod
    def hypotenuse(
        cls,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
    ) -> np.ndarray:
        """
        Compute radial distance on the xy-plane.

        Parameters
        ----------
        x_coordinates
            np.ndarray, shape (n,) or (h, w). Cartesian x coordinates.
        y_coordinates
            np.ndarray, shape (n,) or (h, w). Cartesian y coordinates.
            Stacked notation equivalent is xy with shape (n, 2) or (h, w, 2).

        Returns
        -------
        np.ndarray
            shape (n,) or (h, w)
            Euclidean norm sqrt(x^2 + y^2) for each element.
        """
        norm: np.ndarray = np.sqrt(x_coordinates**2 + y_coordinates**2)
        return norm

    @classmethod
    def from_cartesian(
        cls,
        x_coordinates: np.ndarray,
        y_coordinates: np.ndarray,
        z_coordinates: np.ndarray,
    ) -> SphericalCoordinates:
        """
        Build spherical coordinates from Cartesian coordinates.

        Parameters
        ----------
        x_coordinates
            np.ndarray, shape (n,) or (h, w). Cartesian x coordinates.
        y_coordinates
            np.ndarray, shape (n,) or (h, w). Cartesian y coordinates.
        z_coordinates
            np.ndarray, shape (n,) or (h, w). Cartesian z coordinates.
            Stacked notation equivalent is xyz with shape (n, 3) or (h, w, 3).

        Returns
        -------
        SphericalCoordinates
            longitude and latitude are np.ndarray with shape (n,) or (h, w), in
            radians.
        """
        longitude = np.arctan2(y_coordinates, x_coordinates)
        latitude = np.arctan2(
            z_coordinates,
            cls.hypotenuse(x_coordinates=x_coordinates, y_coordinates=y_coordinates),
        )
        return cls(longitude=longitude, latitude=latitude)
