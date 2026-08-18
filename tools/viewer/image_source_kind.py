from __future__ import annotations

from enum import Enum


class ImageSourceKind(Enum):
    """
    Where the currently viewed image comes from.

    Attributes
    ----------
    SAMPLE_LIBRARY: ImageSourceKind
        An image bundled under the repository's `data/` directory.
    UPLOADED_FILE: ImageSourceKind
        An image uploaded by the user through the browser.
    """

    SAMPLE_LIBRARY = "Sample data"
    UPLOADED_FILE = "Upload"
