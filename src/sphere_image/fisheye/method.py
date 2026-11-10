from __future__ import annotations

from enum import Enum

import numpy as np
from units import Angle


class FisheyeProjectionMethod(Enum):
    """
    Method of fisheye projection.

    Attributes
    ----------
    ORTHOGRAPHIC: FisheyeProjectionMethod
        Orthographic projection method.
    EQUIDISTANT: FisheyeProjectionMethod
        Equidistant projection method.
    STEREOGRAPHIC: FisheyeProjectionMethod
        Stereographic projection method.
    EQUISOLID: FisheyeProjectionMethod
        Equisolid projection method.
    """

    ORTHOGRAPHIC = "Orthographic"
    EQUIDISTANT = "Equidistant"
    STEREOGRAPHIC = "Stereographic"
    EQUISOLID = "Equisolid"

    def calculate_radius(
        self,
        f: float,
        angle: Angle,
    ) -> np.ndarray:
        """
        Normalized radial coordinate on the fisheye image for each incident angle.

        All angles below are in radians. θ is the angle from the optical axis; f is half the
        circular camera FoV. Each law maps θ to r so that r = 1 at the fisheye rim when θ = f.

        Parameters
        ----------
        f: float
            Half camera field of view in radians.
        angle: Angle
            Incident angles.

        Returns
        -------
        np.ndarray
            Dimensionless radius in [0, 1] for θ in [0, f].
        """
        theta = np.asarray(angle.radian, dtype=np.float64)
        match self:
            case FisheyeProjectionMethod.EQUIDISTANT:
                return theta / f
            case FisheyeProjectionMethod.ORTHOGRAPHIC:
                radius: np.ndarray = np.sin(theta) / np.sin(f)
                return radius
            case FisheyeProjectionMethod.STEREOGRAPHIC:
                radius = np.tan(theta / 2) / np.tan(f / 2)
                return radius
            case FisheyeProjectionMethod.EQUISOLID:
                radius = np.sin(theta / 2) / np.sin(f / 2)
                return radius
