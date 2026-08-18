from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LoadedImage:
    """
    An image loaded into the viewer, together with its display name.

    Attributes
    ----------
    pixels: np.ndarray
        Decoded BGR image, shape (H, W, 3).
    display_name: str
        Name shown in the UI, e.g. the source file name.
    """

    pixels: np.ndarray
    display_name: str
