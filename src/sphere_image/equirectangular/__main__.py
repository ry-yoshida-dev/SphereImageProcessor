from __future__ import annotations

import argparse

import cv2
import numpy as np
from handedness_rotation import (
    CoordinateHandedness,
    EulerAngles,
    IntrinsicRotationOrder,
    RotationMatrix,
)
from units import AngleUnit

from .method import EquirectangularProjectionMethod
from .parameter import EquirectangularProcessorParameters


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remap an equirectangular image to another projection.",
    )
    parser.add_argument(
        "-i",
        "--input",
        default="image/equirectangular.JPG",
        help="Path to the input equirectangular image.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tmp_equirectangular.jpg",
        help="Path to write the output image.",
    )
    parser.add_argument(
        "--camera-pointing-up",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override camera vertical orientation (default: EquirectangularProcessorParameters).",
    )
    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=0.0,
        help="Yaw angle in degrees (+: look right).",
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        default=0.0,
        help="Pitch angle in degrees (+: look up).",
    )
    parser.add_argument(
        "--roll-deg",
        type=float,
        default=0.0,
        help="Roll angle in degrees (+: clockwise image tilt).",
    )
    parser.add_argument(
        "--method",
        choices=tuple(
            method.name.lower() for method in EquirectangularProjectionMethod
        ),
        default=EquirectangularProjectionMethod.PERSPECTIVE.name.lower(),
        help="Projection method to run.",
    )
    return parser


def main() -> None:
    parser = build_cli_parser()
    args = parser.parse_args()
    image = cv2.imread(args.input)
    if image is None:
        raise ValueError(f"Failed to read image: {args.input}")

    method = EquirectangularProjectionMethod[args.method.upper()]
    params = EquirectangularProcessorParameters(method=method)
    if args.camera_pointing_up is not None:
        params.is_camera_pointing_up = args.camera_pointing_up

    # Intrinsic ZYX; [-pitch, -roll] on Y/X matches CLI (+pitch look up, +roll).
    euler_angles = EulerAngles(
        value=np.array(
            [args.yaw_deg, -args.pitch_deg, -args.roll_deg],
            dtype=np.float64,
        ),
        order=IntrinsicRotationOrder.ZYX,
        unit=AngleUnit.DEGREE,
    )
    rotation_matrix = RotationMatrix(
        value=euler_angles.rotation_matrix,
        coordinate_handedness=CoordinateHandedness.RIGHT,
    )
    processor = params.build_processor(image=image)
    output_image = processor.run_pipeline(rotation_matrix=rotation_matrix)
    if not cv2.imwrite(args.output, output_image):
        raise RuntimeError(f"Failed to write image: {args.output}")
    print(f"saved {args.method}: {args.output}")


if __name__ == "__main__":
    main()
