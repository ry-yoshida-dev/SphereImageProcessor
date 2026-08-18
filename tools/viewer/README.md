# viewer

## Overview

Streamlit-based interactive viewer for `sphere_image`. A user picks a
fisheye or equirectangular image, and Python converts it into a single
equirectangular texture. That texture is handed to an embedded WebGL
panorama viewer (Three.js): looking around (drag) and zooming (scroll) then
run entirely client-side, with no Streamlit rerun per interaction. This
package is an application built on top of the `sphere_image` library; it is
not part of the installed package and is only ever driven through
`SphereViewerApp` from the repository's `main.py`.

## Components

| Component | Description |
| --- | --- |
| [sphere_viewer_app.py](./sphere_viewer_app.py) | `SphereViewerApp`, the Streamlit page: sidebar controls, texture caching, and wiring into `PanoramaViewer`. |
| [panorama_viewer.py](./panorama_viewer.py) | `PanoramaViewer`, renders the Three.js panorama scene (sphere + texture, drag-to-look, scroll-to-zoom, client-side PNG download) via `st.iframe`. |
| [projection_settings.py](./projection_settings.py) | `ProjectionSettings`, builds the equirectangular texture for the selected `ImageKind` via `match`-`case`. |
| [equirect_texture_builder.py](./equirect_texture_builder.py) | `FisheyeEquirectTextureBuilder`, converts a circular fisheye image into a full equirectangular texture by inverting `FisheyeProjectionMethod.calculate_radius` directly, with no intermediate perspective renders. |
| [image_kind.py](./image_kind.py) | `ImageKind` enum: `FISHEYE` or `EQUIRECTANGULAR`. |
| [image_source_kind.py](./image_source_kind.py) | `ImageSourceKind` enum: `SAMPLE_LIBRARY` or `UPLOADED_FILE`. |
| [sample_image_library.py](./sample_image_library.py) | `SampleImageLibrary`, lists image files bundled under the repository's `data/` directory. |
| [image_loader.py](./image_loader.py) | `ImageLoader`, decodes an image from a file path or from uploaded bytes into a `LoadedImage`. |
| [loaded_image.py](./loaded_image.py) | `LoadedImage`, a decoded image paired with its display name. |

## Notes

- Only `EquirectangularProjectionMethod.PERSPECTIVE` is implemented
  upstream; since equirectangular sources are now used as textures directly
  (no perspective crop is rendered in Python), this restriction doesn't
  apply to the viewer at all.
- The fisheye-to-equirectangular conversion treats every texture pixel as
  the center ray of an imaginary perspective view aimed at that pixel's
  longitude/latitude, so the incident-angle math is identical to
  `FisheyeProcessor`'s own per-pixel computation — verified against
  `FisheyeProcessor`'s perspective renders at multiple yaw/pitch angles.
- The disk's center always maps to the texture's zenith (top); a camera
  mounted pointing downward is handled by flipping the finished texture
  vertically (`is_camera_pointing_up=False`), so its captured content ends
  up at the nadir (bottom) instead.
- The panorama viewer loads Three.js from a CDN (`unpkg.com`); an internet
  connection is required in the browser running the viewer.
