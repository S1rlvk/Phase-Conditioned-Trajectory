#!/usr/bin/env python3
"""
Track hand centroid across frames in Build.ai video.

Uses MediaPipe HandLandmarker. Hand centroid = palm center (palm_x, palm_y).
When occluded, propagates last valid hand position. Reuses logic from extract_kinematics.py.
"""

import argparse
import csv
import os

import cv2
import mediapipe as mp
import numpy as np

PALM_LANDMARK_IDS = [0, 5, 9, 13, 17]
_DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def create_hand_landmarker(model_path):
    """Create a MediaPipe HandLandmarker (Tasks API, >=0.10)."""
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hand(frame_bgr, landmarker):
    """Run MediaPipe HandLandmarker on a BGR frame.

    Returns dict with palm_x, palm_y (centroid) or None if no hand detected.
    Prefers the Right hand; falls back to Left.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    best_idx = 0
    for idx, handedness in enumerate(result.handedness):
        if handedness[0].category_name == "Right":
            best_idx = idx
            break

    landmarks = result.hand_landmarks[best_idx]
    palm_xs = [landmarks[i].x for i in PALM_LANDMARK_IDS]
    palm_ys = [landmarks[i].y for i in PALM_LANDMARK_IDS]

    return {
        "palm_x": float(np.mean(palm_xs)),
        "palm_y": float(np.mean(palm_ys)),
    }


def sample_frames(video_path, sample_fps=0.25):
    """Yield (frame_bgr, timestamp) for each sample time."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    sample_interval = 1.0 / sample_fps
    t = 0.0

    while t < duration:
        t_rounded = round(t, 2)
        cap.set(cv2.CAP_PROP_POS_MSEC, t_rounded * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, t_rounded
        t += sample_interval

    cap.release()


def track_hand_centroid(video_path, output_dir, sample_fps=0.25, model_path=None):
    """Track hand centroid across frames, save to CSV.

    Args:
        video_path: Path to input video.
        output_dir: Output directory for hand_centroids.csv.
        sample_fps: Frame sampling rate.
        model_path: Path to MediaPipe `hand_landmarker.task` (default: next to this module).

    Returns:
        n_frames: Number of frames processed.
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}_hand_centroids.csv")

    mp_path = model_path or _DEFAULT_MODEL
    landmarker = create_hand_landmarker(mp_path)
    last_valid_hand = None
    rows = []

    for frame, timestamp in sample_frames(video_path, sample_fps):
        hand = detect_hand(frame, landmarker)

        if hand is not None:
            last_valid_hand = hand
            centroid_x = hand["palm_x"]
            centroid_y = hand["palm_y"]
            hand_detected = 1
        else:
            if last_valid_hand is not None:
                centroid_x = last_valid_hand["palm_x"]
                centroid_y = last_valid_hand["palm_y"]
            else:
                centroid_x = 0.5
                centroid_y = 0.5
            hand_detected = 0

        rows.append({
            "timestamp": timestamp,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "hand_detected": hand_detected,
        })

    landmarker.close()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "centroid_x", "centroid_y", "hand_detected"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), output_path


def main():
    parser = argparse.ArgumentParser(description="Track hand centroid across frames in Build.ai video")
    parser.add_argument("--video_path", required=True, help="Path to input video file")
    parser.add_argument("--output_dir", required=True, help="Output directory for hand_centroids.csv")
    parser.add_argument("--sample_fps", type=float, default=0.25, help="Frame sampling rate (default: 0.25)")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to hand_landmarker.task (MediaPipe); defaults next to this package",
    )
    args = parser.parse_args()

    print(f"Video:      {args.video_path}")
    print(f"Output:     {args.output_dir}")
    print(f"Sample FPS: {args.sample_fps}")
    print()

    n_frames, output_path = track_hand_centroid(
        args.video_path,
        args.output_dir,
        sample_fps=args.sample_fps,
        model_path=args.model,
    )
    print(f"Tracked {n_frames} frames -> {output_path}")


if __name__ == "__main__":
    main()
