# Examples

Run these from the **repository root** after `pip install -e .` (or set `PYTHONPATH=src`). Clone: `https://github.com/S1rlvk/Phase-Conditioned-Trajectory`.

## 1. Hand trajectory comparison (raw vs standard Kalman vs phase-conditioned)

Requires per-video `factory012_worker017_{id}_kinematics.csv` and `*_annotations.csv` under `--data_dir`.

```bash
python -m phase_conditioned_trajectory.trajectory_pipeline \
  --data_dir /path/to/factory012_worker017_part00 \
  --video_ids 00000 00001 00002 00003 00004 \
  --output_dir results/trajectory_comparison
```

## 2. Next-phase prediction (semantic vs kinematic MLPs)

Requires `*_annotations.csv` and `*_kinematics.npy` files for each video ID listed in `train_next_phase.py`.

```bash
python -m phase_conditioned_trajectory.train_next_phase --data_dir /path/to/data
```

## 3. Phase-conditioned depth filtering

```bash
python -m phase_conditioned_trajectory.phase_depth_filter --help
```

## 4. Hand centroid tracking (MediaPipe)

Download `hand_landmarker.task` from the [MediaPipe hand landmarker model page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and pass `--model` or place it next to the package as documented in `track_hand_centroid.py`.

```bash
python -m phase_conditioned_trajectory.track_hand_centroid \
  --video_path /path/to/video.mp4 \
  --output_dir /path/to/out \
  --model /path/to/hand_landmarker.task
```
