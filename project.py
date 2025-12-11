"""
Motion capture and analysis toolkit for a pendulum video.

The script tracks a dark pendulum bob against a bright background, extracts the
centroid trajectory, estimates the pivot and length, computes angular kinematics,
estimates damping, and simulates a linear damped pendulum for comparison. It can
also write an annotated video overlay that shows the detected centroid, path,
pivot, and simulated trajectory.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Gravitational acceleration (m/s^2)
G = 9.81


def _largest_dark_region_centroid(frame: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return centroid of the largest dark region in a frame.

    A Gaussian blur reduces noise, Otsu thresholding isolates the dark object, and
    the largest contour by area is selected. Returns ``None`` when no valid
    contour is found (e.g., the bob is off-screen).
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Invert threshold: dark bob becomes white foreground.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:  # ignore tiny noise blobs
        return None

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None
    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    return (cx, cy)


def _fit_circle(points: np.ndarray) -> Tuple[Tuple[float, float], float]:
    """Fit a circle to points using the Kåsa method.

    Returns the center ``(x0, y0)`` and radius ``r`` that minimize squared error.
    """

    # Formulate Ax = b for least squares on x0, y0, and r^2.
    x = points[:, 0]
    y = points[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    x0, y0, c = sol
    r = math.sqrt(c + x0 ** 2 + y0 ** 2)
    return (float(x0), float(y0)), float(r)


def _compute_angles(points: np.ndarray, pivot: Tuple[float, float]) -> np.ndarray:
    """Convert 2D positions to angles from vertical (radians)."""

    px, py = pivot
    dx = points[:, 0] - px
    dy = py - points[:, 1]
    return np.arctan2(dx, dy)


def _differentiate(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Compute first derivative with central differences."""

    deriv = np.gradient(values, times)
    return deriv


def _find_peaks(values: np.ndarray) -> List[int]:
    """Return indices of local maxima in a 1D array.

    A peak is a point greater than its immediate neighbors. The first and last
    samples are excluded.
    """

    peaks: List[int] = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            peaks.append(i)
    return peaks


def _estimate_damping(theta: np.ndarray, times: np.ndarray, length: float) -> Tuple[float, float]:
    """Estimate damping ratio and exponential decay rate from angle peaks."""

    peak_indices = _find_peaks(theta)
    if len(peak_indices) < 2:
        return 0.0, 0.0

    peak_times = times[peak_indices]
    peak_amplitudes = np.abs(theta[peak_indices])
    # Fit log envelope: A * exp(-alpha t)
    slope, intercept = np.polyfit(peak_times, np.log(peak_amplitudes), 1)
    alpha = -slope
    omega0 = math.sqrt(G / length)
    zeta = alpha / omega0 if omega0 > 0 else 0.0
    return zeta, alpha


@dataclass
class TrackingResult:
    times: np.ndarray
    centroids: np.ndarray
    pivot: Tuple[float, float]
    length_pixels: float
    angles: np.ndarray
    angular_velocity: np.ndarray


@dataclass
class SimulationResult:
    times: np.ndarray
    angles: np.ndarray


def track_pendulum(
    video_path: Path,
    output_overlay: Optional[Path] = None,
    draw_simulation: Optional[SimulationResult] = None,
) -> TrackingResult:
    """Track pendulum motion in a video and optionally save an overlayed copy."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    dt = 1.0 / fps

    frames: List[np.ndarray] = []
    centroids: List[Tuple[float, float]] = []

    writer = None
    if output_overlay:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_overlay),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        centroid = _largest_dark_region_centroid(frame)
        if centroid is None:
            centroids.append((np.nan, np.nan))
        else:
            centroids.append(centroid)

        if writer is not None:
            if draw_simulation is not None and frame_idx < len(draw_simulation.angles):
                # Draw simulated bob location relative to pivot later after pivot is known.
                pass
            cv2.circle(frame, (int(centroids[-1][0]), int(centroids[-1][1])), 6, (0, 0, 255), -1)
            frames.append(frame)
        frame_idx += 1

    cap.release()

    times = np.arange(len(centroids)) * dt
    centroid_array = np.array(centroids, dtype=float)
    valid = ~np.isnan(centroid_array).any(axis=1)
    if valid.sum() < 3:
        raise RuntimeError("Insufficient valid detections to estimate pendulum parameters.")

    pivot, radius = _fit_circle(centroid_array[valid])
    angles = _compute_angles(centroid_array[:, :2], pivot)
    angular_velocity = _differentiate(angles, times)

    # Write overlay video if requested, now that pivot is known.
    if writer is not None:
        trail: List[Tuple[int, int]] = []
        for idx, frame in enumerate(frames):
            c = centroids[idx]
            if not math.isnan(c[0]):
                trail.append((int(c[0]), int(c[1])))
            cv2.circle(frame, (int(pivot[0]), int(pivot[1])), 5, (0, 255, 0), -1)
            for p1, p2 in zip(trail[:-1], trail[1:]):
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

            if draw_simulation is not None and idx < len(draw_simulation.times):
                sim_angle = draw_simulation.angles[idx]
                sim_x = pivot[0] + radius * math.sin(sim_angle)
                sim_y = pivot[1] + radius * math.cos(sim_angle)
                cv2.circle(frame, (int(sim_x), int(sim_y)), 5, (0, 255, 255), -1)
                cv2.line(frame, (int(pivot[0]), int(pivot[1])), (int(sim_x), int(sim_y)), (0, 255, 255), 2)

            cv2.putText(
                frame,
                f"theta={math.degrees(angles[idx]):.1f} deg",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
            writer.write(frame)
        writer.release()

    return TrackingResult(
        times=times,
        centroids=centroid_array,
        pivot=pivot,
        length_pixels=radius,
        angles=angles,
        angular_velocity=angular_velocity,
    )


def simulate_linear_pendulum(
    times: np.ndarray,
    length: float,
    zeta: float,
    theta0: float,
    theta_dot0: float,
) -> SimulationResult:
    """Simulate a damped small-angle pendulum using RK4 integration."""

    omega0 = math.sqrt(G / length)

    def derivatives(state: np.ndarray) -> np.ndarray:
        theta, theta_dot = state
        theta_ddot = -omega0 ** 2 * theta - 2 * zeta * omega0 * theta_dot
        return np.array([theta_dot, theta_ddot])

    dt = np.gradient(times)
    theta = np.zeros_like(times)
    theta_dot = np.zeros_like(times)
    theta[0] = theta0
    theta_dot[0] = theta_dot0

    for i in range(len(times) - 1):
        h = dt[i]
        state = np.array([theta[i], theta_dot[i]])
        k1 = derivatives(state)
        k2 = derivatives(state + 0.5 * h * k1)
        k3 = derivatives(state + 0.5 * h * k2)
        k4 = derivatives(state + h * k3)
        state_next = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        theta[i + 1], theta_dot[i + 1] = state_next

    return SimulationResult(times=times, angles=theta)


def process_video(
    video_path: Path,
    overlay_path: Optional[Path] = None,
) -> Tuple[TrackingResult, SimulationResult, Tuple[float, float]]:
    """Full pipeline: track video, estimate damping, and simulate."""

    tracking = track_pendulum(video_path)
    zeta, alpha = _estimate_damping(tracking.angles, tracking.times, tracking.length_pixels)

    sim = simulate_linear_pendulum(
        times=tracking.times,
        length=tracking.length_pixels,
        zeta=zeta,
        theta0=tracking.angles[0],
        theta_dot0=tracking.angular_velocity[0],
    )

    if overlay_path:
        # Redraw overlay including simulation results.
        tracking = track_pendulum(video_path, overlay_path, draw_simulation=sim)

    return tracking, sim, (zeta, alpha)


def save_results(
    tracking: TrackingResult,
    sim: SimulationResult,
    damping: Tuple[float, float],
    output_dir: Path,
) -> None:
    """Save motion data and summary metrics to CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.column_stack(
        [
            tracking.times,
            tracking.centroids[:, 0],
            tracking.centroids[:, 1],
            tracking.angles,
            tracking.angular_velocity,
            sim.angles,
        ]
    )
    header = ",".join(
        [
            "time_s",
            "centroid_x_px",
            "centroid_y_px",
            "theta_rad",
            "theta_dot_rad_s",
            "sim_theta_rad",
        ]
    )
    np.savetxt(output_dir / "trajectory.csv", data, delimiter=",", header=header)

    zeta, alpha = damping
    summary_lines = [
        f"Pivot (px): {tracking.pivot}",
        f"Pendulum length (px): {tracking.length_pixels:.3f}",
        f"Damping ratio: {zeta:.5f}",
        f"Decay rate (1/s): {alpha:.5f}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Pendulum motion capture and analysis")
    parser.add_argument("video", type=Path, help="Path to the pendulum video file")
    parser.add_argument("--overlay", type=Path, help="Path to write annotated video (mp4)")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Directory for CSV and summary")
    args = parser.parse_args(argv)

    tracking, sim, damping = process_video(args.video, args.overlay)
    save_results(tracking, sim, damping, args.output)
    print(f"Detected pivot (px): {tracking.pivot}")
    print(f"Estimated pendulum length (px): {tracking.length_pixels:.3f}")
    print(f"Damping ratio: {damping[0]:.5f}")
    if args.overlay:
        print(f"Overlay video saved to {args.overlay}")
    print(f"Results saved in {args.output}")


if __name__ == "__main__":
    main()
