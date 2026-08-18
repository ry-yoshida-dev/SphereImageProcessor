from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .loaded_image import LoadedImage


class ImageLoader:
    """
    Decodes viewer input images from disk or from in-memory bytes.
    """

    @staticmethod
    def load_from_path(path: Path) -> LoadedImage:
        """
        Load an image from a file path.

        Parameters
        ----------
        path: Path
            Path to a BGR-decodable image file.

        Returns
        -------
        LoadedImage
            The decoded image, named after `path`.

        Raises
        ------
        ValueError
            If the file cannot be decoded as an image.
        """
        pixels = cv2.imread(str(path))
        if pixels is None:
            raise ValueError(f"Failed to read image: {path}")
        return LoadedImage(pixels=pixels, display_name=path.name)

    @staticmethod
    def load_from_bytes(data: bytes, display_name: str) -> LoadedImage:
        """
        Load an image from an in-memory byte buffer.

        Parameters
        ----------
        data: bytes
            Encoded image bytes, e.g. from an uploaded file.
        display_name: str
            Name shown in the UI for the decoded image.

        Returns
        -------
        LoadedImage
            The decoded image, named `display_name`.

        Raises
        ------
        ValueError
            If the bytes cannot be decoded as an image.
        """
        encoded_buffer = np.frombuffer(data, dtype=np.uint8)
        pixels = cv2.imdecode(encoded_buffer, cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError(f"Failed to decode uploaded image: {display_name}")
        return LoadedImage(pixels=pixels, display_name=display_name)
