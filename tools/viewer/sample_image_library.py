from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_SAMPLE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class SampleImageLibrary:
    """
    Catalog of bundled sample images available for the viewer.

    Attributes
    ----------
    directory: Path
        Directory scanned for sample images.
    """

    directory: Path

    def list_image_paths(self) -> list[Path]:
        """
        List sample image files in `directory`, sorted by file name.

        Returns
        -------
        list[Path]
            Paths whose suffix matches a supported image extension. Empty if
            `directory` does not exist.
        """
        if not self.directory.is_dir():
            return []
        return sorted(
            path
            for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() in _SAMPLE_IMAGE_SUFFIXES
        )
