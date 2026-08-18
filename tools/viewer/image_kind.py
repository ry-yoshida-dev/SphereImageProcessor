from __future__ import annotations

from enum import Enum


class ImageKind(Enum):
    """
    Source image projection family shown in the viewer.

    Attributes
    ----------
    FISHEYE: ImageKind
        Circular fisheye image, remapped with `FisheyeProcessor`.
    EQUIRECTANGULAR: ImageKind
        Full 360-degree equirectangular image, remapped with `EquirectangularProcessor`.
    """

    FISHEYE = "Fisheye (180°)"
    EQUIRECTANGULAR = "Equirectangular (360°)"

    @property
    def display_name(self) -> str:
        """
        Human-readable label shown in the UI.

        Returns
        -------
        str
            The label text.
        """
        return self.value
