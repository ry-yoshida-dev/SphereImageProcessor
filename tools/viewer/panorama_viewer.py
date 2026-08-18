from __future__ import annotations

import base64

import cv2
import numpy as np
import streamlit as st

from sphere_image.fisheye import FisheyeProjectionMethod

from .image_kind import ImageKind

_THREE_JS_CDN_URL = "https://unpkg.com/three@0.128.0/build/three.min.js"

_HTML_TEMPLATE = """
<div id="panorama-toolbar" style="max-width:__MAX_WIDTH_PX__px;margin:0 auto 6px;
     display:flex;align-items:center;justify-content:space-between;gap:0.75rem;">
  <button id="panorama-download-button" type="button"
    style="padding:0.35rem 0.75rem;border:1px solid #999;border-radius:0.375rem;
           background:#f0f2f6;cursor:pointer;font-size:0.875rem;">
    Download current view
  </button>
  <span id="panorama-source-coordinate" style="font-size:0.8rem;font-family:monospace;
        color:#555;white-space:nowrap;">Hover the view to read source coordinates</span>
</div>
<div id="panorama-container" style="width:100%;max-width:__MAX_WIDTH_PX__px;margin:0 auto;
     aspect-ratio:__ASPECT_RATIO__;background:#000;border-radius:0.375rem;overflow:hidden;
     cursor:crosshair;"></div>
<script src="__THREE_JS_CDN_URL__"></script>
<script>
(function () {
  const container = document.getElementById("panorama-container");
  const downloadButton = document.getElementById("panorama-download-button");
  const sourceCoordinateLabel = document.getElementById("panorama-source-coordinate");

  const IMAGE_KIND = "__IMAGE_KIND__";
  const FISHEYE_METHOD = "__FISHEYE_METHOD__";
  const CAMERA_FOV_DEGREE = __CAMERA_FOV_DEGREE__;
  const IS_CAMERA_POINTING_UP = __IS_CAMERA_POINTING_UP__;
  const SOURCE_IMAGE_WIDTH = __SOURCE_IMAGE_WIDTH__;
  const SOURCE_IMAGE_HEIGHT = __SOURCE_IMAGE_HEIGHT__;

  const camera = new THREE.PerspectiveCamera(
    __INITIAL_FOV_DEGREE__,
    container.clientWidth / container.clientHeight,
    1,
    1100
  );
  camera.position.set(0, 0, 0.1);

  const scene = new THREE.Scene();
  const geometry = new THREE.SphereGeometry(500, 60, 40);
  geometry.scale(-1, 1, 1);
  const texture = new THREE.TextureLoader().load(
    "data:image/jpeg;base64,__TEXTURE_BASE64__"
  );
  const material = new THREE.MeshBasicMaterial({ map: texture });
  const sphereMesh = new THREE.Mesh(geometry, material);
  scene.add(sphereMesh);

  const raycaster = new THREE.Raycaster();
  const pointerNdc = new THREE.Vector2();
  const sphereBoundary = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 500);
  const sphereIntersectionPoint = new THREE.Vector3();

  function calculateFisheyeRadius(halfFovRadian, incidentAngleRadian) {
    switch (FISHEYE_METHOD) {
      case "Orthographic":
        return Math.sin(incidentAngleRadian) / Math.sin(halfFovRadian);
      case "Stereographic":
        return Math.tan(incidentAngleRadian / 2) / Math.tan(halfFovRadian / 2);
      case "Equisolid":
        return Math.sin(incidentAngleRadian / 2) / Math.sin(halfFovRadian / 2);
      default:
        return incidentAngleRadian / halfFovRadian;
    }
  }

  function computeSourceCoordinate(intersectionPoint) {
    const polarAngleRadian = Math.acos(
      THREE.MathUtils.clamp(intersectionPoint.y / intersectionPoint.length(), -1, 1)
    );
    const azimuthRadian = Math.atan2(intersectionPoint.z, intersectionPoint.x);
    const longitudeRadian = azimuthRadian;
    const latitudeRadian = Math.PI / 2 - polarAngleRadian;

    const normalizedU = (longitudeRadian + Math.PI) / (2 * Math.PI);
    const normalizedVDisplayed = (Math.PI / 2 - latitudeRadian) / Math.PI;

    if (IMAGE_KIND !== "Fisheye (180°)") {
      return {
        x: normalizedU * SOURCE_IMAGE_WIDTH,
        y: normalizedVDisplayed * SOURCE_IMAGE_HEIGHT,
      };
    }

    const normalizedVBeforeFlip = IS_CAMERA_POINTING_UP
      ? normalizedVDisplayed
      : 1 - normalizedVDisplayed;
    const incidentAngleRadian = normalizedVBeforeFlip * Math.PI;
    const halfFovRadian = THREE.MathUtils.degToRad(CAMERA_FOV_DEGREE) / 2;
    if (incidentAngleRadian > halfFovRadian) {
      return null;
    }

    const longitudeBeforeFlip = normalizedU * 2 * Math.PI - Math.PI;
    const radius = calculateFisheyeRadius(halfFovRadian, incidentAngleRadian);
    const normalizedImageU = 0.5 + 0.5 * radius * Math.cos(longitudeBeforeFlip);
    const normalizedImageV = 0.5 + 0.5 * radius * Math.sin(longitudeBeforeFlip);
    const sourceX = normalizedImageU * (SOURCE_IMAGE_WIDTH - 1);
    const sourceY = normalizedImageV * (SOURCE_IMAGE_HEIGHT - 1);
    if (sourceX < 0 || sourceX > SOURCE_IMAGE_WIDTH - 1 || sourceY < 0 || sourceY > SOURCE_IMAGE_HEIGHT - 1) {
      return null;
    }
    return { x: sourceX, y: sourceY };
  }

  function updateSourceCoordinateLabel(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointerNdc.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    raycaster.setFromCamera(pointerNdc, camera);
    const hitPoint = raycaster.ray.intersectSphere(sphereBoundary, sphereIntersectionPoint);
    if (hitPoint === null) {
      sourceCoordinateLabel.textContent = "Hover the view to read source coordinates";
      return;
    }
    const sourceCoordinate = computeSourceCoordinate(hitPoint);
    sourceCoordinateLabel.textContent =
      sourceCoordinate === null
        ? "Outside the source image"
        : `Source pixel: (${Math.round(sourceCoordinate.x)}, ${Math.round(sourceCoordinate.y)})`;
  }

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  let longitudeDegree = 0;
  let latitudeDegree = 0;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragStartLongitudeDegree = 0;
  let dragStartLatitudeDegree = 0;

  function onPointerDown(event) {
    isDragging = true;
    dragStartX = event.clientX;
    dragStartY = event.clientY;
    dragStartLongitudeDegree = longitudeDegree;
    dragStartLatitudeDegree = latitudeDegree;
    renderer.domElement.setPointerCapture(event.pointerId);
    renderer.domElement.style.cursor = "grabbing";
  }

  function onPointerMove(event) {
    if (isDragging) {
      longitudeDegree = (dragStartX - event.clientX) * 0.1 + dragStartLongitudeDegree;
      latitudeDegree = (event.clientY - dragStartY) * 0.1 + dragStartLatitudeDegree;
    }
    updateSourceCoordinateLabel(event);
  }

  function onPointerUp() {
    isDragging = false;
    renderer.domElement.style.cursor = "crosshair";
  }

  function onPointerLeave() {
    sourceCoordinateLabel.textContent = "Hover the view to read source coordinates";
  }

  function onWheel(event) {
    event.preventDefault();
    const nextFov = camera.fov + event.deltaY * 0.02;
    camera.fov = Math.min(100, Math.max(20, nextFov));
    camera.updateProjectionMatrix();
  }

  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointermove", onPointerMove);
  renderer.domElement.addEventListener("pointerup", onPointerUp);
  renderer.domElement.addEventListener("pointercancel", onPointerUp);
  renderer.domElement.addEventListener("pointerleave", onPointerLeave);
  renderer.domElement.addEventListener("wheel", onWheel, { passive: false });

  new ResizeObserver(function () {
    if (container.clientWidth === 0 || container.clientHeight === 0) return;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }).observe(container);

  downloadButton.addEventListener("click", function () {
    renderer.render(scene, camera);
    const link = document.createElement("a");
    link.download = "sphere_view.png";
    link.href = renderer.domElement.toDataURL("image/png");
    link.click();
  });

  function animate() {
    requestAnimationFrame(animate);
    latitudeDegree = Math.max(-85, Math.min(85, latitudeDegree));
    const phi = THREE.MathUtils.degToRad(90 - latitudeDegree);
    const theta = THREE.MathUtils.degToRad(longitudeDegree);
    const lookAtX = 500 * Math.sin(phi) * Math.cos(theta);
    const lookAtY = 500 * Math.cos(phi);
    const lookAtZ = 500 * Math.sin(phi) * Math.sin(theta);
    camera.lookAt(lookAtX, lookAtY, lookAtZ);
    renderer.render(scene, camera);
  }
  animate();
})();
</script>
"""


class PanoramaViewer:
    """
    Client-side WebGL panorama viewer embedded via `st.iframe`.

    The equirectangular texture is generated once in Python and handed to
    the browser; looking around (drag) and zooming (scroll) then run
    entirely in JavaScript at the display's own frame rate, with no
    Streamlit rerun.
    """

    @staticmethod
    def render(
        texture_bgr: np.ndarray,
        initial_fov_degree: float,
        aspect_ratio: float,
        max_width_px: int,
        image_kind: ImageKind,
        fisheye_method: FisheyeProjectionMethod,
        camera_fov_degree: float,
        is_camera_pointing_up: bool,
        source_image_width: int,
        source_image_height: int,
    ) -> None:
        """
        Encode `texture_bgr` and render the panorama viewer component.

        Parameters
        ----------
        texture_bgr: np.ndarray
            Equirectangular texture, in BGR channel order.
        initial_fov_degree: float
            Camera field of view, in degrees, when the viewer first loads.
        aspect_ratio: float
            Width divided by height of the rendered viewer canvas. The
            canvas fills the available width (up to `max_width_px`) and
            derives its height from this ratio (via CSS `aspect-ratio`), so
            it stays well-proportioned regardless of the browser's window
            width.
        max_width_px: int
            Maximum width of the rendered viewer canvas, in pixels. Caps how
            large the canvas grows on wide windows/columns.
        image_kind: ImageKind
            Source image projection family, used to invert the cursor's
            sphere position back into source-image coordinates.
        fisheye_method: FisheyeProjectionMethod
            Fisheye radius equation to invert. Only used when `image_kind`
            is `FISHEYE`.
        camera_fov_degree: float
            Circular fisheye field of view, in degrees. Only used when
            `image_kind` is `FISHEYE`.
        is_camera_pointing_up: bool
            Whether the camera is mounted pointing upward. Only used when
            `image_kind` is `FISHEYE`.
        source_image_width: int
            Width, in pixels, of the original source image.
        source_image_height: int
            Height, in pixels, of the original source image.

        Raises
        ------
        ValueError
            If `texture_bgr` cannot be encoded as JPEG.
        """
        is_encoded, encoded_texture = cv2.imencode(
            ".jpg", texture_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not is_encoded:
            raise ValueError("Failed to encode the panorama texture as JPEG.")
        texture_base64 = base64.b64encode(encoded_texture.tobytes()).decode("ascii")

        html = (
            _HTML_TEMPLATE.replace("__THREE_JS_CDN_URL__", _THREE_JS_CDN_URL)
            .replace("__TEXTURE_BASE64__", texture_base64)
            .replace("__INITIAL_FOV_DEGREE__", str(initial_fov_degree))
            .replace("__ASPECT_RATIO__", str(aspect_ratio))
            .replace("__MAX_WIDTH_PX__", str(max_width_px))
            .replace("__IMAGE_KIND__", image_kind.value)
            .replace("__FISHEYE_METHOD__", fisheye_method.value)
            .replace("__CAMERA_FOV_DEGREE__", str(camera_fov_degree))
            .replace("__IS_CAMERA_POINTING_UP__", "true" if is_camera_pointing_up else "false")
            .replace("__SOURCE_IMAGE_WIDTH__", str(source_image_width))
            .replace("__SOURCE_IMAGE_HEIGHT__", str(source_image_height))
        )
        st.iframe(html, height="content", width="stretch")
