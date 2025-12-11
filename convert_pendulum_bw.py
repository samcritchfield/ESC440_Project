"""Convert a pendulum video to black-and-white frames using OpenCV."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def convert_to_black_and_white(input_path: Path, output_path: Path) -> None:
    """Read ``input_path`` video, convert frames to grayscale, and write ``output_path``."""

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        writer.write(gray)

    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pendulum video to black and white.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("pendulum.mp4"),
        help="Path to the input pendulum video (default: pendulum.mp4)",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("pendulum_bw.mp4"),
        help="Path to save the black-and-white video (default: pendulum_bw.mp4)",
    )
    args = parser.parse_args()

    convert_to_black_and_white(args.input, args.output)


if __name__ == "__main__":
    main()
