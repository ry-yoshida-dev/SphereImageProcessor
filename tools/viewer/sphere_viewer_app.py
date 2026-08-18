from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st

from sphere_image.fisheye import FisheyeProjectionMethod

from .image_kind import ImageKind
from .image_loader import ImageLoader
from .image_source_kind import ImageSourceKind
from .loaded_image import LoadedImage
from .panorama_viewer import PanoramaViewer
from .projection_settings import ProjectionSettings
from .sample_image_library import SampleImageLibrary

_ASPECT_RATIO_OPTIONS = {
    "16:9": 16 / 9,
    "4:3": 4 / 3,
    "1:1": 1.0,
    "9:16": 9 / 16,
}
_DEFAULT_ASPECT_RATIO_LABEL = "16:9"


class SphereViewerApp:
    """
    Streamlit page for interactively viewing fisheye and equirectangular images.

    An equirectangular texture is generated in Python whenever the source
    image or projection settings change. Looking around and zooming then run
    entirely client-side in an embedded WebGL panorama viewer, with no
    server round-trip per interaction.
    """

    def __init__(self, sample_image_directory: Path | None = None) -> None:
        """
        Parameters
        ----------
        sample_image_directory: Path | None
            Directory scanned for bundled sample images. Defaults to the
            repository's `data/` directory.
        """
        self._sample_image_library = SampleImageLibrary(
            directory=sample_image_directory or self._default_sample_image_directory()
        )

    def run(self) -> None:
        """
        Render the full viewer page for the current Streamlit session.
        """
        st.set_page_config(page_title="Sphere Image Viewer", layout="wide")
        st.title("Sphere Image Viewer")

        with st.sidebar:
            loaded_image = self._render_image_source_controls()
            image_kind = self._render_image_kind_control()
            fisheye_method, camera_fov_degree, is_camera_pointing_up = (
                self._render_fisheye_controls(image_kind)
            )
            texture_width = self._render_quality_control()
            initial_fov_degree = st.slider(
                "Initial zoom (FoV, deg)", min_value=30.0, max_value=110.0, value=80.0, step=1.0
            )
            aspect_ratio = self._render_aspect_ratio_control()
            max_width_px = st.slider(
                "Viewer width (px)", min_value=320, max_value=1600, value=1600, step=20
            )
            st.caption("Drag to look around, scroll to zoom — fully interactive, no reload.")

        if loaded_image is None:
            st.info("Select or upload an image to begin.")
            return

        settings = ProjectionSettings(
            image_kind=image_kind,
            fisheye_method=fisheye_method,
            camera_fov_degree=camera_fov_degree,
            is_camera_pointing_up=is_camera_pointing_up,
            texture_width=texture_width,
            texture_height=texture_width // 2,
        )
        with st.spinner("Building panorama texture..."):
            texture = self._build_equirect_texture(loaded_image, settings)
        st.caption(loaded_image.display_name)
        PanoramaViewer.render(
            texture_bgr=texture,
            initial_fov_degree=initial_fov_degree,
            aspect_ratio=aspect_ratio,
            max_width_px=max_width_px,
            image_kind=image_kind,
            fisheye_method=fisheye_method,
            camera_fov_degree=camera_fov_degree,
            is_camera_pointing_up=is_camera_pointing_up,
            source_image_width=loaded_image.pixels.shape[1],
            source_image_height=loaded_image.pixels.shape[0],
        )

    @staticmethod
    @st.cache_data(show_spinner=False)
    def _build_equirect_texture(
        loaded_image: LoadedImage, settings: ProjectionSettings
    ) -> np.ndarray:
        """
        Build and cache the equirectangular texture for the given inputs.

        Parameters
        ----------
        loaded_image: LoadedImage
            The currently selected source image.
        settings: ProjectionSettings
            Projection settings selected in the sidebar.

        Returns
        -------
        np.ndarray
            The equirectangular texture, in BGR channel order.
        """
        texture: np.ndarray = settings.build_equirect_texture(image=loaded_image.pixels)
        return texture

    def _render_image_source_controls(self) -> LoadedImage | None:
        """
        Render source-selection widgets and return the currently loaded image.

        Returns
        -------
        LoadedImage | None
            The loaded image, or `None` if nothing is selected yet.
        """
        st.header("Image")
        image_source_kind = st.radio(
            "Source",
            options=list(ImageSourceKind),
            format_func=lambda kind: kind.value,
            horizontal=True,
        )
        match image_source_kind:
            case ImageSourceKind.UPLOADED_FILE:
                uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
                if uploaded_file is None:
                    return None
                uploaded_image: LoadedImage = self._load_uploaded_image(
                    data=uploaded_file.getvalue(), display_name=uploaded_file.name
                )
                return uploaded_image
            case _:
                sample_paths = self._sample_image_library.list_image_paths()
                if not sample_paths:
                    st.warning(
                        f"No sample images found under {self._sample_image_library.directory}."
                    )
                    return None
                selected_path = st.selectbox(
                    "Sample image", options=sample_paths, format_func=lambda path: path.name
                )
                sample_image: LoadedImage = self._load_sample_image(selected_path)
                return sample_image

    @staticmethod
    @st.cache_data(show_spinner=False)
    def _load_sample_image(path: Path) -> LoadedImage:
        """
        Load and cache a sample image, keyed by its path.

        Parameters
        ----------
        path: Path
            Path to the sample image file.

        Returns
        -------
        LoadedImage
            The decoded image.
        """
        return ImageLoader.load_from_path(path)

    @staticmethod
    @st.cache_data(show_spinner=False)
    def _load_uploaded_image(data: bytes, display_name: str) -> LoadedImage:
        """
        Load and cache an uploaded image, keyed by its bytes and name.

        Parameters
        ----------
        data: bytes
            The uploaded file's raw bytes.
        display_name: str
            Name shown in the UI for the decoded image.

        Returns
        -------
        LoadedImage
            The decoded image.
        """
        return ImageLoader.load_from_bytes(data=data, display_name=display_name)

    @staticmethod
    def _render_image_kind_control() -> ImageKind:
        """
        Render the projection-family selector.

        Returns
        -------
        ImageKind
            The selected image kind.
        """
        st.header("Projection")
        image_kind = st.radio(
            "Image kind",
            options=list(ImageKind),
            format_func=lambda kind: kind.display_name,
            horizontal=True,
        )
        assert image_kind is not None
        return image_kind

    @staticmethod
    def _render_fisheye_controls(
        image_kind: ImageKind,
    ) -> tuple[FisheyeProjectionMethod, float, bool]:
        """
        Render fisheye-only controls, when applicable.

        Parameters
        ----------
        image_kind: ImageKind
            Currently selected image kind.

        Returns
        -------
        tuple[FisheyeProjectionMethod, float, bool]
            The selected fisheye method, camera field of view in degrees,
            and whether the camera is mounted pointing upward. Defaults are
            returned unused when `image_kind` is not `FISHEYE`.
        """
        if image_kind is not ImageKind.FISHEYE:
            st.caption("Equirectangular sources are used as-is; no conversion is needed.")
            return FisheyeProjectionMethod.EQUIDISTANT, 185.0, True

        fisheye_methods = list(FisheyeProjectionMethod)
        fisheye_method = st.selectbox(
            "Fisheye method",
            options=fisheye_methods,
            index=fisheye_methods.index(FisheyeProjectionMethod.EQUIDISTANT),
            format_func=lambda method: method.value,
        )
        camera_fov_degree = st.slider(
            "Camera FoV (deg)", min_value=60.0, max_value=359.0, value=185.0, step=1.0
        )
        is_camera_pointing_up = st.checkbox("Camera pointing up", value=True)
        return fisheye_method, camera_fov_degree, is_camera_pointing_up

    @staticmethod
    def _render_quality_control() -> int:
        """
        Render the texture-quality selector.

        Returns
        -------
        int
            Equirectangular texture width, in pixels (height is width / 2).
        """
        st.header("Quality")
        texture_width_options = {
            "Low (1024px)": 1024,
            "Medium (2048px)": 2048,
            "High (3072px)": 3072,
        }
        selected_label = st.select_slider(
            "Texture resolution",
            options=list(texture_width_options.keys()),
            value="Medium (2048px)",
        )
        return texture_width_options[selected_label]

    @staticmethod
    def _render_aspect_ratio_control() -> float:
        """
        Render the viewer aspect-ratio selector.

        Returns
        -------
        float
            Width divided by height for the panorama viewer canvas.
        """
        selected_label = st.selectbox(
            "Viewer aspect ratio",
            options=list(_ASPECT_RATIO_OPTIONS.keys()),
            index=list(_ASPECT_RATIO_OPTIONS.keys()).index(_DEFAULT_ASPECT_RATIO_LABEL),
        )
        return _ASPECT_RATIO_OPTIONS[selected_label]

    @staticmethod
    def _default_sample_image_directory() -> Path:
        """
        Resolve the repository's `data/` directory relative to this file.

        Returns
        -------
        Path
            The default sample image directory.
        """
        repository_root = Path(__file__).resolve().parents[2]
        return repository_root / "data"
