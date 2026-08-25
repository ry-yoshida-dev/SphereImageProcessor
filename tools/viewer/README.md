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

The panorama viewer's own toolbar has an "Extract this view" button: it
sends the camera's current yaw/pitch/field of view back to Python, where
they are combined with sidebar-entered fields (camera name, view name,
roll, output size, output FoV basis) and rendered below the viewer as a
YAML snippet the user can copy into `SphereImageCalibration`'s
`flat_views.yaml`.

## Components

| Component | Description |
| --- | --- |
| [sphere_viewer_app.py](./sphere_viewer_app.py) | `SphereViewerApp`, the Streamlit page: sidebar controls, texture caching, and wiring into `PanoramaViewer`. |
| [panorama_viewer.py](./panorama_viewer.py) | `PanoramaViewer`, renders the panorama viewer as a bidirectional Streamlit component (see [frontend/](./frontend/)) and returns the last `CapturedView`, if any. |
| [frontend/](./frontend/) | Static (build-free) frontend for the `PanoramaViewer` component: the Three.js panorama scene (drag-to-look, scroll-to-zoom, client-side PNG download, "Extract this view") plus the hand-rolled Streamlit component postMessage protocol. |
| [captured_view.py](./captured_view.py) | `CapturedView`, the yaw/pitch/field of view captured from the panorama viewer's camera. |
| [flat_view_export_fields.py](./flat_view_export_fields.py) | `FlatViewExportFields`, the sidebar-entered `flat_views.yaml` fields other than yaw/pitch/FoV (camera name, view name, roll, output size, output FoV basis). |
| [flat_view_export_settings.py](./flat_view_export_settings.py) | `FlatViewExportSettings`, `FlatViewExportFields` combined with a `CapturedView`, rendered as a `flat_views.yaml`-formatted YAML snippet. |
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
- `PanoramaViewer.render`'s `key` argument must stay constant across
  reruns: `declare_component` derives a component's identity from its
  arguments when no explicit `key` is given, and the texture (among other
  args) changes on nearly every rerun, which would otherwise discard the
  last captured view.
