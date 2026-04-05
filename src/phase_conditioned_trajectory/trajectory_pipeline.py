#!/usr/bin/env python3
"""
Hand Trajectory Stabilization Pipeline
=======================================
3-way comparison: Raw | FidelityDepth (standard Kalman) | Phase-Conditioned Kalman

Pipeline:
  Video → Hand Tracking → Raw XY Trajectory
                ↓
      ┌─────────┴─────────┐
      ↓                   ↓
  Kalman Filter      Phase Detector
  (FidelityDepth)    (Annotations)
      ↓                   ↓
  Reconstructed       Phase Labels
  Trajectory              ↓
      └─────────┬─────────┘
                ↓
      Phase-Conditioned Filter
                ↓
      Stabilized Trajectory

Usage:
  python trajectory_pipeline.py --data_dir factory012_worker017_part00 \
      --video_ids 00000 00001 00002 00003 00004 \
      --output_dir results/trajectory_comparison
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

# ─────────────────────────────────────────────
#  Phase-specific XY trajectory priors
# ─────────────────────────────────────────────
# State: [x, y, vx, vy]  |  Measurement: [x, y]
# Q_vel: process noise on velocity (how much acceleration we allow)
# R_pos: measurement noise (how much we trust MediaPipe)

PHASE_TRAJECTORY_PRIORS = {
    "IDLE": {
        "Q_vel": 5e-5,   # Near-stationary between task cycles
        "R_pos": 0.06,   # Reduced trust: IDLE detections can be noisy/wandering
    },
    "REACH": {
        "Q_vel": 4e-3,   # Purposeful fast movement toward target
        "R_pos": 0.015,  # Trust measurement: deliberate, detectable motion
    },
    "ALIGN": {
        "Q_vel": 8e-4,   # Slow fine-grained positioning
        "R_pos": 0.010,  # High trust: precise voluntary movement
    },
    "INSERT": {
        "Q_vel": 5e-5,   # Near-stationary at object
        "R_pos": 0.008,  # Highest trust: stable contact
    },
    "ROTATE": {
        "Q_vel": 1e-4,   # Slight rotational repositioning
        "R_pos": 0.012,  # Good trust: controlled movement
    },
    "WITHDRAW": {
        "Q_vel": 4e-3,   # Fast purposeful withdrawal
        "R_pos": 0.015,  # Trust measurement: deliberate motion
    },
}

# FidelityDepth (standard) — single fixed noise across all phases
FIDELITY_Q_VEL = 1.5e-3   # Moderate: works reasonably everywhere
FIDELITY_R_POS = 0.025    # Moderate measurement trust

PHASE_COLORS = {
    "IDLE":     "#f0f0f0",
    "REACH":    "#ffe0e0",
    "ALIGN":    "#fff4cc",
    "INSERT":   "#d6eaff",
    "ROTATE":   "#ead6ff",
    "WITHDRAW": "#d6ffd6",
}


# ─────────────────────────────────────────────
#  Kalman Filter Factory
# ─────────────────────────────────────────────

def make_kalman_2d(dt: float, x0: float, y0: float,
                   q_pos: float, q_vel: float, r_pos: float) -> KalmanFilter:
    """Create a 2D constant-velocity Kalman filter.

    State:       [x, y, vx, vy]
    Measurement: [x, y]
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State transition: position integrates velocity
    kf.F = np.array([
        [1, 0, dt,  0],
        [0, 1,  0, dt],
        [0, 0,  1,  0],
        [0, 0,  0,  1],
    ], dtype=float)

    # Measurement matrix: observe [x, y] only
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    # Initial state
    kf.x = np.array([x0, y0, 0.0, 0.0])

    # Initial covariance
    kf.P = np.diag([0.01, 0.01, 0.1, 0.1])

    # Process noise: small position uncertainty, larger velocity uncertainty
    kf.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

    # Measurement noise
    kf.R = np.diag([r_pos, r_pos])

    return kf


# ─────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────

def load_kinematics(kin_path: Path) -> pd.DataFrame:
    """Load kinematics CSV → DataFrame with [timestamp, palm_x, palm_y, hand_detected]."""
    df = pd.read_csv(kin_path)
    required = ["timestamp", "palm_x", "palm_y", "hand_detected"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {kin_path}")
    return df[required].copy()


def load_phase_labels(annot_path: Path, timestamps: np.ndarray) -> list[str]:
    """Map annotation segments to per-timestamp phase labels."""
    segments = []
    annot_df = pd.read_csv(annot_path)
    for _, row in annot_df.iterrows():
        segments.append({
            "start": float(row["start_time"]),
            "end":   float(row["end_time"]),
            "phase": str(row["phase"]).strip().upper(),
        })

    labels = []
    for t in timestamps:
        phase = "IDLE"
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                phase = seg["phase"]
                break
        labels.append(phase)
    return labels


# ─────────────────────────────────────────────
#  Filtering
# ─────────────────────────────────────────────

def fidelity_depth_filter(df: pd.DataFrame, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Standard constant-velocity Kalman filter (FidelityDepth approach).

    Uses uniform noise parameters — no phase information.
    Returns (filtered_x, filtered_y).
    """
    x0 = float(df["palm_x"].iloc[0])
    y0 = float(df["palm_y"].iloc[0])

    kf = make_kalman_2d(dt, x0, y0,
                        q_pos=1e-4, q_vel=FIDELITY_Q_VEL, r_pos=FIDELITY_R_POS)

    xs, ys = [], []
    for _, row in df.iterrows():
        kf.predict()
        if row["hand_detected"] == 1:
            kf.update(np.array([row["palm_x"], row["palm_y"]]))
        xs.append(kf.x[0])
        ys.append(kf.x[1])

    return np.array(xs), np.array(ys)


def phase_conditioned_filter(df: pd.DataFrame, phases: list[str],
                              dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Phase-conditioned Kalman filter.

    Noise parameters switch per phase:
    - Stable phases (INSERT, IDLE): tight process noise → very smooth
    - Active phases (REACH, WITHDRAW): loose process noise → tracks fast movement

    Returns (filtered_x, filtered_y).
    """
    x0 = float(df["palm_x"].iloc[0])
    y0 = float(df["palm_y"].iloc[0])

    # Start with IDLE prior
    priors = PHASE_TRAJECTORY_PRIORS["IDLE"]
    kf = make_kalman_2d(dt, x0, y0,
                        q_pos=1e-4, q_vel=priors["Q_vel"], r_pos=priors["R_pos"])

    current_phase = None
    xs, ys = [], []

    for (_, row), phase in zip(df.iterrows(), phases):
        # Switch noise params when phase changes
        if phase != current_phase:
            p = PHASE_TRAJECTORY_PRIORS.get(phase, PHASE_TRAJECTORY_PRIORS["IDLE"])
            kf.Q = np.diag([1e-4, 1e-4, p["Q_vel"], p["Q_vel"]])
            kf.R = np.diag([p["R_pos"], p["R_pos"]])
            current_phase = phase

        kf.predict()
        if row["hand_detected"] == 1:
            kf.update(np.array([row["palm_x"], row["palm_y"]]))
        xs.append(kf.x[0])
        ys.append(kf.x[1])

    return np.array(xs), np.array(ys)


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────

def jitter(traj: np.ndarray) -> float:
    """Mean absolute frame-to-frame displacement."""
    return float(np.mean(np.abs(np.diff(traj))))


def variance_of_diff(traj: np.ndarray) -> float:
    return float(np.var(np.diff(traj)))


def compute_metrics(raw_x, raw_y, fd_x, fd_y, pc_x, pc_y,
                    phases: list[str]) -> dict:
    """Compute 3-way comparison metrics overall and per phase."""

    def traj_metrics(x, y, label):
        jx, jy = jitter(x), jitter(y)
        return {
            f"{label}_jitter_x": jx,
            f"{label}_jitter_y": jy,
            f"{label}_jitter_xy": (jx + jy) / 2,
            f"{label}_var_x": variance_of_diff(x),
            f"{label}_var_y": variance_of_diff(y),
        }

    metrics = {}
    metrics.update(traj_metrics(raw_x, raw_y, "raw"))
    metrics.update(traj_metrics(fd_x, fd_y, "fidelity"))
    metrics.update(traj_metrics(pc_x, pc_y, "phase_cond"))

    # Reduction percentages (vs raw)
    raw_j = metrics["raw_jitter_xy"]
    fd_j  = metrics["fidelity_jitter_xy"]
    pc_j  = metrics["phase_cond_jitter_xy"]
    metrics["fidelity_jitter_reduction_pct"] = (1 - fd_j / raw_j) * 100 if raw_j > 0 else 0.0
    metrics["phase_cond_jitter_reduction_pct"] = (1 - pc_j / raw_j) * 100 if raw_j > 0 else 0.0
    metrics["phase_cond_vs_fidelity_improvement_pct"] = (1 - pc_j / fd_j) * 100 if fd_j > 0 else 0.0

    # Per-phase jitter reduction
    phases_arr = np.array(phases)
    phase_metrics = {}
    for ph in PHASE_TRAJECTORY_PRIORS:
        mask = phases_arr == ph
        if mask.sum() < 2:
            continue
        # Need consecutive indices for diff — use mask on full array then diff
        idxs = np.where(mask)[0]
        # Only take consecutive pairs within the same phase
        consec = idxs[np.diff(idxs, prepend=idxs[0]-1) == 1]
        if len(consec) < 2:
            continue
        r_j = (jitter(raw_x[consec]) + jitter(raw_y[consec])) / 2
        f_j = (jitter(fd_x[consec]) + jitter(fd_y[consec])) / 2
        p_j = (jitter(pc_x[consec]) + jitter(pc_y[consec])) / 2
        if r_j > 0:
            phase_metrics[ph] = {
                "raw_jitter": round(r_j, 6),
                "fidelity_jitter": round(f_j, 6),
                "phase_cond_jitter": round(p_j, 6),
                "fidelity_reduction_pct": round((1 - f_j / r_j) * 100, 1),
                "phase_cond_reduction_pct": round((1 - p_j / r_j) * 100, 1),
            }

    metrics["per_phase"] = phase_metrics
    return metrics


# ─────────────────────────────────────────────
#  Visualizations
# ─────────────────────────────────────────────

def plot_trajectory_comparison(timestamps, raw_x, raw_y,
                                fd_x, fd_y, pc_x, pc_y,
                                phases, video_id, output_path):
    """3-way trajectory comparison with phase-band background."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Hand Trajectory Stabilization — {video_id}",
                 fontsize=14, fontweight="bold")

    for ax, raw, fd, pc, ylabel, title in [
        (axes[0], raw_x, fd_x, pc_x, "X Position (normalized)", "X Trajectory"),
        (axes[1], raw_y, fd_y, pc_y, "Y Position (normalized)", "Y Trajectory"),
    ]:
        # Phase background bands
        phase_arr = np.array(phases)
        for i in range(len(phase_arr) - 1):
            color = PHASE_COLORS.get(phase_arr[i], "#ffffff")
            ax.axvspan(timestamps[i], timestamps[i + 1], facecolor=color, alpha=0.5)
        # Last span
        ax.axvspan(timestamps[-2], timestamps[-1],
                   facecolor=PHASE_COLORS.get(phase_arr[-1], "#ffffff"), alpha=0.5)

        ax.plot(timestamps, raw, color="#aaaaaa", linewidth=1.2, alpha=0.8,
                label="Raw (MediaPipe)", zorder=2)
        ax.plot(timestamps, fd, color="#2196F3", linewidth=1.8,
                label="FidelityDepth (Standard Kalman)", zorder=3)
        ax.plot(timestamps, pc, color="#E91E63", linewidth=2.0,
                label="Phase-Conditioned Kalman", zorder=4)

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_ylim([-0.05, 1.05])

    axes[1].set_xlabel("Time (seconds)", fontsize=11)

    # Phase legend
    legend_patches = [
        mpatches.Patch(facecolor=PHASE_COLORS[ph], alpha=0.6, label=ph)
        for ph in PHASE_COLORS
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6,
               fontsize=9, title="Phase", bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_2d_trajectory(raw_x, raw_y, fd_x, fd_y, pc_x, pc_y,
                       phases, video_id, output_path):
    """2D spatial trajectory paths comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"2D Hand Trajectory Paths — {video_id}",
                 fontsize=13, fontweight="bold")

    phase_arr = np.array(phases)
    phase_list = list(PHASE_COLORS.keys())

    def scatter_phase(ax, xs, ys, title):
        for ph in phase_list:
            mask = phase_arr == ph
            if mask.sum() == 0:
                continue
            ax.scatter(xs[mask], ys[mask], c=PHASE_COLORS[ph],
                       edgecolors="gray", s=40, linewidths=0.4,
                       label=ph, zorder=3)
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.8, zorder=2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X (normalized)")
        ax.set_ylabel("Y (normalized)")
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.invert_yaxis()  # Image coordinates: y increases downward
        ax.grid(alpha=0.3, linestyle="--")

    scatter_phase(axes[0], raw_x, raw_y,   "Raw (MediaPipe)")
    scatter_phase(axes[1], fd_x,  fd_y,    "FidelityDepth")
    scatter_phase(axes[2], pc_x,  pc_y,    "Phase-Conditioned")

    handles = [mpatches.Patch(facecolor=PHASE_COLORS[ph], label=ph,
                               edgecolor="gray") for ph in phase_list]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_phase_reduction(per_phase_metrics, video_id, output_path):
    """Bar chart: jitter reduction per phase for FidelityDepth vs Phase-Conditioned."""
    phases = [ph for ph in PHASE_TRAJECTORY_PRIORS if ph in per_phase_metrics]
    if not phases:
        return

    fd_reductions = [per_phase_metrics[ph]["fidelity_reduction_pct"] for ph in phases]
    pc_reductions = [per_phase_metrics[ph]["phase_cond_reduction_pct"] for ph in phases]

    x = np.arange(len(phases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_fd = ax.bar(x - width / 2, fd_reductions, width,
                     label="FidelityDepth", color="#2196F3", alpha=0.85)
    bars_pc = ax.bar(x + width / 2, pc_reductions, width,
                     label="Phase-Conditioned", color="#E91E63", alpha=0.85)

    ax.set_xlabel("Phase", fontsize=12)
    ax.set_ylabel("Jitter Reduction vs Raw (%)", fontsize=12)
    ax.set_title(f"Per-Phase Jitter Reduction — {video_id}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=11)
    ax.legend(fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim([-10, 100])

    # Value labels
    for bar in bars_fd:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8.5, color="#1565C0")
    for bar in bars_pc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=8.5, color="#880E4F")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_multi_video_summary(all_metrics: list[dict], video_ids: list[str], output_path):
    """Aggregate jitter reduction across all videos."""
    fd_reductions = [m["fidelity_jitter_reduction_pct"] for m in all_metrics]
    pc_reductions = [m["phase_cond_jitter_reduction_pct"] for m in all_metrics]
    pc_vs_fd = [m["phase_cond_vs_fidelity_improvement_pct"] for m in all_metrics]

    x = np.arange(len(video_ids))
    width = 0.28

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, fd_reductions, width, label="FidelityDepth vs Raw",
           color="#2196F3", alpha=0.85)
    ax.bar(x,         pc_reductions, width, label="Phase-Conditioned vs Raw",
           color="#E91E63", alpha=0.85)
    ax.bar(x + width, pc_vs_fd,      width, label="Phase-Cond vs FidelityDepth",
           color="#4CAF50", alpha=0.85)

    ax.set_xlabel("Video ID", fontsize=12)
    ax.set_ylabel("Jitter Reduction (%)", fontsize=12)
    ax.set_title("Multi-Video Jitter Reduction: 3-Way Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(video_ids, fontsize=10, rotation=20, ha="right")
    ax.legend(fontsize=10)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3, linestyle="--", axis="y")

    # Mean line annotations
    ax.axhline(np.mean(fd_reductions), color="#1565C0", linestyle="--", linewidth=1,
               label=f"FD mean: {np.mean(fd_reductions):.0f}%")
    ax.axhline(np.mean(pc_reductions), color="#880E4F", linestyle="--", linewidth=1,
               label=f"PC mean: {np.mean(pc_reductions):.0f}%")
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────
#  Per-video processing
# ─────────────────────────────────────────────

def process_video(data_dir: Path, video_id: str, output_dir: Path,
                  dt: float = 4.0, full_vis: bool = True) -> dict:
    """Run full pipeline for a single video.

    Returns metrics dict.
    """
    prefix = f"factory012_worker017_{video_id}"
    kin_path    = data_dir / f"{prefix}_kinematics.csv"
    annot_path  = data_dir / f"{prefix}_annotations.csv"

    print(f"\n[{video_id}] Loading data...")
    df = load_kinematics(kin_path)
    timestamps = df["timestamp"].values
    phases = load_phase_labels(annot_path, timestamps)

    raw_x = df["palm_x"].values
    raw_y = df["palm_y"].values

    print(f"[{video_id}] Applying FidelityDepth Kalman filter...")
    fd_x, fd_y = fidelity_depth_filter(df, dt)

    print(f"[{video_id}] Applying Phase-Conditioned Kalman filter...")
    pc_x, pc_y = phase_conditioned_filter(df, phases, dt)

    print(f"[{video_id}] Computing metrics...")
    metrics = compute_metrics(raw_x, raw_y, fd_x, fd_y, pc_x, pc_y, phases)
    metrics["video_id"] = video_id
    metrics["n_frames"] = len(df)
    metrics["occlusion_frames"] = int((df["hand_detected"] == 0).sum())

    # Save trajectories CSV
    traj_df = pd.DataFrame({
        "timestamp": timestamps,
        "phase": phases,
        "raw_x": raw_x,
        "raw_y": raw_y,
        "fidelity_x": fd_x,
        "fidelity_y": fd_y,
        "phase_cond_x": pc_x,
        "phase_cond_y": pc_y,
    })
    traj_df.to_csv(output_dir / f"{video_id}_trajectories.csv", index=False)

    if full_vis:
        print(f"[{video_id}] Generating visualizations...")
        plot_trajectory_comparison(
            timestamps, raw_x, raw_y, fd_x, fd_y, pc_x, pc_y,
            phases, video_id,
            output_dir / f"{video_id}_trajectory_comparison.png"
        )
        plot_2d_trajectory(
            raw_x, raw_y, fd_x, fd_y, pc_x, pc_y,
            phases, video_id,
            output_dir / f"{video_id}_2d_paths.png"
        )
        if metrics.get("per_phase"):
            plot_per_phase_reduction(
                metrics["per_phase"], video_id,
                output_dir / f"{video_id}_per_phase_reduction.png"
            )

    print(f"[{video_id}] Results:")
    print(f"  Raw jitter (XY avg):            {metrics['raw_jitter_xy']:.4f}")
    print(f"  FidelityDepth jitter:           {metrics['fidelity_jitter_xy']:.4f}  "
          f"({metrics['fidelity_jitter_reduction_pct']:.1f}% reduction)")
    print(f"  Phase-Conditioned jitter:       {metrics['phase_cond_jitter_xy']:.4f}  "
          f"({metrics['phase_cond_jitter_reduction_pct']:.1f}% reduction)")
    print(f"  Phase-Cond vs FidelityDepth:    {metrics['phase_cond_vs_fidelity_improvement_pct']:.1f}% better")

    return metrics


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hand Trajectory Stabilization Pipeline: Raw vs FidelityDepth vs Phase-Conditioned"
    )
    parser.add_argument("--data_dir", default="factory012_worker017_part00",
                        help="Directory with kinematics + annotations CSVs")
    parser.add_argument("--video_ids", nargs="+",
                        default=["00000", "00001", "00002", "00003", "00004"],
                        help="Video IDs to process")
    parser.add_argument("--output_dir", default="results/trajectory_comparison",
                        help="Output directory for results + plots")
    parser.add_argument("--dt", type=float, default=4.0,
                        help="Seconds between samples (default: 4.0 for 0.25fps)")
    parser.add_argument("--single", default=None,
                        help="Process only this video ID (full vis + detailed output)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        video_ids = [args.single]
    else:
        video_ids = args.video_ids

    # Verify inputs exist
    for vid in video_ids:
        prefix = f"factory012_worker017_{vid}"
        for fname in [f"{prefix}_kinematics.csv", f"{prefix}_annotations.csv"]:
            if not (data_dir / fname).exists():
                raise FileNotFoundError(f"Missing: {data_dir / fname}")

    all_metrics = []
    for vid in video_ids:
        full_vis = (len(video_ids) == 1) or True  # Always generate per-video plots
        m = process_video(data_dir, vid, output_dir, dt=args.dt, full_vis=full_vis)
        all_metrics.append(m)

    # Multi-video summary
    if len(all_metrics) > 1:
        print("\n" + "=" * 60)
        print("AGGREGATE RESULTS")
        print("=" * 60)
        fd_mean  = np.mean([m["fidelity_jitter_reduction_pct"] for m in all_metrics])
        pc_mean  = np.mean([m["phase_cond_jitter_reduction_pct"] for m in all_metrics])
        imp_mean = np.mean([m["phase_cond_vs_fidelity_improvement_pct"] for m in all_metrics])
        print(f"FidelityDepth jitter reduction:       {fd_mean:.1f}% (mean across {len(video_ids)} videos)")
        print(f"Phase-Conditioned jitter reduction:   {pc_mean:.1f}% (mean)")
        print(f"Phase-Conditioned vs FidelityDepth:   {imp_mean:.1f}% better (mean)")
        print("=" * 60)

        plot_multi_video_summary(all_metrics, video_ids,
                                 output_dir / "multi_video_summary.png")

    # Save all metrics JSON
    metrics_path = output_dir / "all_metrics.json"
    # Serialize: convert numpy types
    def _to_python(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(
        json.dumps(all_metrics, default=_to_python)
    )
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved all metrics → {metrics_path}")
    print("Done. ✓")


if __name__ == "__main__":
    main()
