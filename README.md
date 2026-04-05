# Phase-conditioned trajectory

Research code for **phase-conditioned Kalman filtering** of hand trajectories (2D screen space and depth), plus **next-phase prediction** experiments comparing semantic (phase-label) vs kinematic representations under occlusion-style degradation.

The core idea: assembly phases impose different motion statistics (reach vs insert vs idle). Switching process and measurement noise of a constant-velocity Kalman filter by **discrete phase** yields smoother, more plausible trajectories than a single fixed noise model (a “FidelityDepth-style” baseline), while staying consistent with observed hand points when visible.

## Repository layout

| Path | Contents |
|------|----------|
| `src/phase_conditioned_trajectory/` | Library modules: trajectory pipeline, depth filter, next-phase training, hand tracking |
| `examples/` | Command-line examples for each entry point |
| `docs/technical_report.md` | Full write-up: task, dataset split, semantic vs kinematic gap, degradation study |
| `results/` | Saved **metrics** (JSON) and **trajectory CSVs**; **plots** (PNG) for the multi-video comparison |

## Installation

```bash
git clone https://github.com/S1rlvk/Phase-Conditioned-Trajectory.git
cd Phase-Conditioned-Trajectory
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Dependencies are listed in `requirements.txt` and `pyproject.toml` (PyTorch, MediaPipe, FilterPy, scikit-learn, etc.).

## Modules

### Trajectory pipeline (`trajectory_pipeline.py`)

Compares three tracks per video:

1. **Raw** — MediaPipe (or similar) palm centroid samples  
2. **FidelityDepth-style** — one constant-velocity Kalman model with fixed noise everywhere  
3. **Phase-conditioned** — same structure, but `Q` / `R` depend on the current assembly phase from annotations  

Outputs per-video CSVs, aggregate `all_metrics.json`, and figures (time series, 2D paths, per-phase jitter reduction, multi-video summary).

```bash
python -m phase_conditioned_trajectory.trajectory_pipeline --help
```

### Next-phase prediction (`train_next_phase.py`)

Trains two identical MLPs to predict phase at \(t{+}1\) from either stacked one-hot phases at \(t{-}1, t\) (**semantic**) or stacked kinematic feature vectors (**kinematic**). Includes a test-time **degradation** sweep on the kinematic input. See `docs/technical_report.md` for the reported accuracy gap and interpretation.

### Depth filtering (`phase_depth_filter.py`)

Applies the same phase-switching idea to **scalar depth** trajectories (e.g. MiDaS-style depth) with phase-specific velocity priors.

### Hand centroid tracking (`track_hand_centroid.py`)

Optional preprocessing: sample frames from video, run MediaPipe Hand Landmarker, write centroid CSVs. Requires a `hand_landmarker.task` asset (not shipped; see MediaPipe docs).

## Results in this repo

Under `results/`:

- **`trajectory_comparison/`** — Per-video trajectory CSVs, `all_metrics.json`, and PNG plots (including `multi_video_summary.png`).
- **`phase_depth_filtering/`** — Depth trajectory export and stability metrics.

Reproduce figures by re-running the pipeline with your data paths; committed outputs correspond to the experiment configuration described in `docs/technical_report.md`.

## Data

This repository does **not** include raw factory videos or proprietary assets. File naming in the code follows the `factory012_worker017_*` convention used in development; point `--data_dir` at your own extracted CSV/NPY layout.

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE).

## Citation

If you use this code or the experimental setup in research, please cite the repository and, where relevant, cite your own dataset source. A suggested BibTeX entry:

```bibtex
@software{phase_conditioned_trajectory,
  title        = {Phase-conditioned trajectory: filtering and next-phase prediction for assembly monitoring},
  year         = {2026},
  howpublished = {\url{https://github.com/S1rlvk/Phase-Conditioned-Trajectory}},
}
```
