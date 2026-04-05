#!/usr/bin/env python3
"""
Phase-conditioned depth filtering using Kalman filter.

Takes noisy depth measurements and phase labels, produces stable depth trajectory.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

# Phase-specific depth priors
PHASE_DEPTH_PRIORS = {
    'IDLE': {
        'velocity_mean': 0.0,      # No movement
        'velocity_std': 0.02,      # Low variance
        'measurement_noise': 0.3,  # Moderate trust in MiDaS
    },
    'REACH': {
        'velocity_mean': -0.015,   # Approaching (negative = closer)
        'velocity_std': 0.03,      # Smooth approach
        'measurement_noise': 0.4,  # Less trust (hand moving fast)
    },
    'ALIGN': {
        'velocity_mean': -0.005,   # Still approaching but slower
        'velocity_std': 0.015,     # Very smooth
        'measurement_noise': 0.2,  # More trust (deliberate movement)
    },
    'INSERT': {
        'velocity_mean': 0.0,      # Stable at object
        'velocity_std': 0.01,      # Very stable
        'measurement_noise': 0.15, # High trust (stable contact)
    },
    'ROTATE': {
        'velocity_mean': 0.0,      # Stable at object
        'velocity_std': 0.01,     # Very stable
        'measurement_noise': 0.15, # High trust (stable contact)
    },
    'WITHDRAW': {
        'velocity_mean': 0.015,    # Receding (positive = farther)
        'velocity_std': 0.03,      # Smooth withdrawal
        'measurement_noise': 0.4,  # Less trust (hand moving fast)
    }
}


class PhaseAwareDepthFilter:
    """
    Kalman filter with phase-conditioned process and measurement noise.

    State: [depth, velocity]
    Measurement: raw depth from MiDaS
    """

    def __init__(self, dt=1.0, initial_depth=1.0):
        """
        Initialize Kalman filter.

        Args:
            dt: Time step between measurements (seconds)
            initial_depth: Initial depth estimate (meters)
        """
        # 2D state: [depth, velocity]
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition matrix (depth = depth + velocity * dt)
        self.kf.F = np.array([
            [1., dt],
            [0., 1.]
        ])

        # Measurement matrix (observe depth only)
        self.kf.H = np.array([[1., 0.]])

        # Initial state
        self.kf.x = np.array([initial_depth, 0.0])

        # Initial covariance (high uncertainty)
        self.kf.P = np.array([
            [1.0, 0.0],
            [0.0, 0.1]
        ])

        self.current_phase = None
        self.dt = dt

    def update_phase_priors(self, phase):
        """Update Kalman filter noise parameters based on phase."""
        if phase not in PHASE_DEPTH_PRIORS:
            phase = 'IDLE'  # Default fallback

        priors = PHASE_DEPTH_PRIORS[phase]

        # Process noise (uncertainty in velocity model)
        velocity_variance = priors['velocity_std'] ** 2
        self.kf.Q = np.array([
            [0.001, 0.0],  # Small depth process noise
            [0.0, velocity_variance]
        ])

        # Measurement noise (trust in MiDaS reading)
        self.kf.R = np.array([[priors['measurement_noise'] ** 2]])

        # Expected velocity for this phase
        self.expected_velocity = priors['velocity_mean']

    def predict_and_update(self, phase, raw_depth):
        """
        Kalman filter prediction and update step.

        Args:
            phase: Current manipulation phase
            raw_depth: Raw depth measurement from MiDaS

        Returns:
            filtered_depth: Filtered depth estimate
        """
        # Update priors if phase changed
        if phase != self.current_phase:
            self.update_phase_priors(phase)
            self.current_phase = phase

        # Prediction step
        self.kf.predict()

        # Optionally nudge velocity toward expected value
        # (soft constraint on phase-conditioned velocity)
        velocity_error = self.expected_velocity - self.kf.x[1]
        self.kf.x[1] += 0.1 * velocity_error  # Gentle pull

        # Update step with measurement
        self.kf.update(raw_depth)

        return self.kf.x[0]  # Return filtered depth


def load_merged_data(manifest_path):
    """Load merged data manifest."""
    df = pd.read_csv(manifest_path)

    # Ensure required columns exist
    required = ['timestamp', 'phase', 'centroid_x', 'centroid_y', 'depth_path']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    return df


def extract_hand_depth_trajectory(df):
    """
    Extract hand depth values from depth maps at centroid locations.

    Args:
        df: Merged manifest DataFrame

    Returns:
        raw_depths: List of depth values at hand centroid
    """
    raw_depths = []

    for _, row in df.iterrows():
        depth_path = row['depth_path']
        centroid_x = row['centroid_x']
        centroid_y = row['centroid_y']

        if pd.isna(depth_path) or pd.isna(centroid_x):
            raw_depths.append(np.nan)
            continue

        # Load depth map
        depth_map = np.load(depth_path)

        # Convert normalized centroid to pixel coordinates
        h, w = depth_map.shape
        px = int(centroid_x * w)
        py = int(centroid_y * h)

        # Clip to valid range
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)

        # Extract depth at centroid
        hand_depth = depth_map[py, px]
        raw_depths.append(hand_depth)

    return np.array(raw_depths)


def apply_phase_filtering(df, raw_depths, dt=1.0):
    """
    Apply phase-conditioned Kalman filtering.

    Args:
        df: Merged manifest DataFrame with 'phase' column
        raw_depths: Raw depth measurements
        dt: Time step between frames

    Returns:
        filtered_depths: Filtered depth trajectory
    """
    # Initialize filter with first valid depth
    initial_depth = np.nanmean(raw_depths[:10])  # Average of first few frames
    if np.isnan(initial_depth):
        initial_depth = 1.0  # Fallback if all NaN
    phase_filter = PhaseAwareDepthFilter(dt=dt, initial_depth=initial_depth)

    filtered_depths = []

    for phase, raw_depth in zip(df['phase'], raw_depths):
        if np.isnan(raw_depth):
            # No measurement, just predict
            phase_filter.kf.predict()
            filtered_depths.append(phase_filter.kf.x[0])
        else:
            # Prediction + update
            filtered_depth = phase_filter.predict_and_update(phase, raw_depth)
            filtered_depths.append(filtered_depth)

    return np.array(filtered_depths)


def compute_stability_metrics(raw_depths, filtered_depths):
    """
    Compute temporal stability metrics.

    Args:
        raw_depths: Raw depth trajectory
        filtered_depths: Filtered depth trajectory

    Returns:
        dict with variance, jitter, smoothness metrics
    """
    # Remove NaNs
    valid_mask = ~np.isnan(raw_depths)
    raw_valid = raw_depths[valid_mask]
    filtered_valid = filtered_depths[valid_mask]

    if len(raw_valid) < 2 or len(filtered_valid) < 2:
        return {
            'raw_variance': 0.0,
            'filtered_variance': 0.0,
            'variance_reduction_pct': 0.0,
            'raw_jitter': 0.0,
            'filtered_jitter': 0.0,
            'jitter_reduction_pct': 0.0,
        }

    # Temporal derivative (jitter)
    raw_diff = np.diff(raw_valid)
    filtered_diff = np.diff(filtered_valid)

    # Metrics
    raw_variance = np.var(raw_diff)
    filtered_variance = np.var(filtered_diff)
    variance_reduction = (1 - filtered_variance / raw_variance) * 100 if raw_variance > 0 else 0.0

    raw_jitter = np.mean(np.abs(raw_diff))
    filtered_jitter = np.mean(np.abs(filtered_diff))
    jitter_reduction = (1 - filtered_jitter / raw_jitter) * 100 if raw_jitter > 0 else 0.0

    return {
        'raw_variance': float(raw_variance),
        'filtered_variance': float(filtered_variance),
        'variance_reduction_pct': float(variance_reduction),
        'raw_jitter': float(raw_jitter),
        'filtered_jitter': float(filtered_jitter),
        'jitter_reduction_pct': float(jitter_reduction),
    }


def plot_depth_comparison(df, raw_depths, filtered_depths, output_path):
    """
    Generate comparison plot (Sanskar's style).

    Args:
        df: Merged manifest with 'phase' column
        raw_depths: Raw depth trajectory
        filtered_depths: Filtered depth trajectory
        output_path: Path to save plot
    """
    # Normalize depths for visualization (like Sanskar's plot)
    raw_mean = np.nanmean(raw_depths)
    raw_std = np.nanstd(raw_depths)
    if raw_std == 0:
        raw_std = 1.0

    raw_normalized = (raw_depths - raw_mean) / raw_std + 3.0
    filtered_normalized = (filtered_depths - raw_mean) / raw_std + 3.0

    # Phase colors for background
    phase_colors = {
        'IDLE': '#f5f5f5',
        'REACH': '#ffe6e6',
        'ALIGN': '#fff4e6',
        'INSERT': '#e6f7ff',
        'ROTATE': '#f0e6ff',
        'WITHDRAW': '#e6ffe6'
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Background coloring by phase
    phases = df['phase'].values
    for i in range(len(phases) - 1):
        color = phase_colors.get(phases[i], '#ffffff')
        ax.axvspan(i, i + 1, facecolor=color, alpha=0.3)

    # Plot trajectories
    ax.plot(raw_normalized,
            color='gray',
            alpha=0.7,
            linewidth=1,
            label='MiDaS Raw (Normalized)')

    ax.plot(filtered_normalized,
            color='#ff1493',  # Pink
            linewidth=2,
            label='MiDaS + Phase-Conditioned')

    ax.set_xlabel('Temporal Index (Frames)', fontsize=12)
    ax.set_ylabel('Metric Depth (Scaled Depth)', fontsize=12)
    ax.set_title('Depth Trajectory Stability: Raw vs. Phase-Conditioned',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, 6])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Phase-conditioned depth filtering")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to merged_manifest.csv")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save results")
    parser.add_argument("--dt", type=float, default=1.0,
                        help="Time step between frames (seconds)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading merged data...")
    df = load_merged_data(args.manifest)

    print("Extracting hand depth trajectory...")
    raw_depths = extract_hand_depth_trajectory(df)

    print("Applying phase-conditioned filtering...")
    filtered_depths = apply_phase_filtering(df, raw_depths, dt=args.dt)

    print("Computing stability metrics...")
    metrics = compute_stability_metrics(raw_depths, filtered_depths)

    print("\n" + "="*50)
    print("TEMPORAL STABILITY METRICS")
    print("="*50)
    print(f"Raw variance:        {metrics['raw_variance']:.6f}")
    print(f"Filtered variance:   {metrics['filtered_variance']:.6f}")
    print(f"Variance reduction:  {metrics['variance_reduction_pct']:.1f}%")
    print(f"")
    print(f"Raw jitter:          {metrics['raw_jitter']:.6f}")
    print(f"Filtered jitter:     {metrics['filtered_jitter']:.6f}")
    print(f"Jitter reduction:    {metrics['jitter_reduction_pct']:.1f}%")
    print("="*50 + "\n")

    # Save metrics
    metrics_path = output_dir / "stability_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save trajectories
    trajectories_path = output_dir / "depth_trajectories.csv"
    traj_df = pd.DataFrame({
        'timestamp': df['timestamp'],
        'phase': df['phase'],
        'raw_depth': raw_depths,
        'filtered_depth': filtered_depths
    })
    traj_df.to_csv(trajectories_path, index=False)
    print(f"Saved trajectories to {trajectories_path}")

    # Generate plot
    plot_path = output_dir / "depth_stability_comparison.png"
    print("Generating comparison plot...")
    plot_depth_comparison(df, raw_depths, filtered_depths, plot_path)

    print("\nDone! ✓")


if __name__ == "__main__":
    main()
