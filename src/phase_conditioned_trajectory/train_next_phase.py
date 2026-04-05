#!/usr/bin/env python3
"""
Compare semantic (S_t) vs kinematic (K_t) representations for next-phase
prediction using identical MLPs.

Task: given representations at t-1 and t, predict phase at t+1.
Includes test-time degradation experiment on kinematic model.
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PHASES = ["IDLE", "REACH", "ALIGN", "INSERT", "ROTATE", "WITHDRAW"]
PHASE2IDX = {p: i for i, p in enumerate(PHASES)}
NUM_CLASSES = len(PHASES)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "factory012_worker017_part00")
TRAIN_IDS = [
    "factory012_worker017_00000",
    "factory012_worker017_00001",
    "factory012_worker017_00002",
]
TEST_IDS = [
    "factory012_worker017_00003",
    "factory012_worker017_00004",
]

SAMPLE_FPS = 0.25
EPOCHS = 50
LR = 1e-3
HIDDEN = 32


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_semantic(video_id):
    """Load segment CSV and resolve to per-timestamp phase indices at 0.25fps."""
    csv_path = os.path.join(DATA_DIR, f"{video_id}_annotations.csv")
    segments = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append({
                "start": float(row["start_time"]),
                "end": float(row["end_time"]),
                "phase": row["phase"].strip().upper(),
            })

    duration = segments[-1]["end"]
    interval = 1.0 / SAMPLE_FPS
    timestamps = np.arange(0, duration, interval)

    phase_indices = np.zeros(len(timestamps), dtype=np.int64)
    for i, t in enumerate(timestamps):
        for seg in segments:
            if seg["start"] <= t < seg["end"]:
                phase_indices[i] = PHASE2IDX[seg["phase"]]
                break
    return phase_indices, timestamps


def load_kinematics(video_id):
    """Load kinematic features NPY, shape (N, 10) or (N, 13) with object_x, object_y, object_confidence."""
    npy_path = os.path.join(DATA_DIR, f"{video_id}_kinematics.npy")
    return np.load(npy_path).astype(np.float32)


def build_dataset(video_ids):
    """Build paired (X_K, X_S, y) arrays from a list of videos."""
    all_xk, all_xs, all_y = [], [], []

    for vid in video_ids:
        S, _ = load_semantic(vid)
        K = load_kinematics(vid)
        N = min(len(S), len(K))
        S, K = S[:N], K[:N]

        for t in range(1, N - 1):
            xk = np.concatenate([K[t - 1], K[t]])
            s_prev = np.zeros(NUM_CLASSES, dtype=np.float32)
            s_curr = np.zeros(NUM_CLASSES, dtype=np.float32)
            s_prev[S[t - 1]] = 1.0
            s_curr[S[t]] = 1.0
            xs = np.concatenate([s_prev, s_curr])
            y = S[t + 1]

            all_xk.append(xk)
            all_xs.append(xs)
            all_y.append(y)

    return (
        torch.tensor(np.stack(all_xk)),
        torch.tensor(np.stack(all_xs)),
        torch.tensor(np.array(all_y, dtype=np.int64)),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model, X, y, epochs=EPOCHS, lr=LR, class_weights=None):
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        logits = model(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"    epoch {epoch+1:3d}  loss={loss.item():.4f}  train_acc={acc:.3f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, label=""):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1).numpy()
    y_np = y.numpy()
    acc = accuracy_score(y_np, preds)
    print(f"\n--- {label} ---")
    print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(classification_report(y_np, preds, target_names=PHASES, zero_division=0))
    cm = confusion_matrix(y_np, preds, labels=list(range(NUM_CLASSES)))
    print("Confusion matrix (rows=true, cols=pred):")
    header = "        " + " ".join(f"{p:>6s}" for p in PHASES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {PHASES[i]:>6s} " + " ".join(f"{v:6d}" for v in row))
    return acc


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

def degrade_kinematics(X, level):
    """Test-time corruption of kinematic features.

    X shape: (N, 20) or (N, 26) = concat(K[t-1], K[t]).
    10-D half: [0:7] position/delta, [7:9] flow, [9] hand_detected.
    13-D half adds [10:13] object_x, object_y, object_confidence.
    """
    X = X.clone()
    d = X.shape[1]
    half = d // 2
    pos_end = 7
    flow_end = 9

    if level == 1:
        X[:, 0:pos_end] += torch.randn_like(X[:, 0:pos_end]) * 1.0
        X[:, half : half + pos_end] += torch.randn_like(X[:, half : half + pos_end]) * 1.0
    elif level == 2:
        X[:, 0:pos_end] = 0
        X[:, half : half + pos_end] = 0
    elif level == 3:
        X[:, 0:flow_end] = 0
        X[:, half : half + flow_end] = 0
    return X


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def print_class_distribution(y, label):
    counts = np.bincount(y.numpy(), minlength=NUM_CLASSES)
    total = counts.sum()
    print(f"\n{label} class distribution ({total} samples):")
    for i, phase in enumerate(PHASES):
        pct = counts[i] / total * 100 if total > 0 else 0
        print(f"  {phase:>10s}: {counts[i]:4d} ({pct:5.1f}%)")


def compute_class_weights(y):
    counts = np.bincount(y.numpy(), minlength=NUM_CLASSES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.tensor(weights)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global DATA_DIR
    parser = argparse.ArgumentParser(
        description="Next-phase prediction: semantic vs kinematic MLPs"
    )
    parser.add_argument(
        "--data_dir",
        default=DATA_DIR,
        help="Directory with *_annotations.csv and *_kinematics.npy per video",
    )
    args = parser.parse_args()
    DATA_DIR = os.path.abspath(args.data_dir)

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("NEXT-PHASE PREDICTION: Semantic vs Kinematic")
    print("=" * 60)

    # --- Load data ---
    print("\n[1] Loading data...")
    X_train_K, X_train_S, y_train = build_dataset(TRAIN_IDS)
    X_test_K, X_test_S, y_test = build_dataset(TEST_IDS)

    print(f"  Train: {len(y_train)} samples from {len(TRAIN_IDS)} videos")
    print(f"  Test:  {len(y_test)} samples from {len(TEST_IDS)} videos")
    print(f"  X_K shape: {X_train_K.shape}  X_S shape: {X_train_S.shape}")

    # --- Class distribution ---
    print_class_distribution(y_train, "Train")
    print_class_distribution(y_test, "Test")

    class_weights = compute_class_weights(y_train)
    print(f"\n  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # --- Normalize kinematics ---
    print("\n[2] Normalizing kinematic features (train stats)...")
    mean_K = X_train_K.mean(dim=0)
    std_K = X_train_K.std(dim=0) + 1e-6
    X_train_K = (X_train_K - mean_K) / std_K
    X_test_K = (X_test_K - mean_K) / std_K

    # --- Train semantic model ---
    print("\n[3] Training SEMANTIC model...")
    model_S = MLP(input_dim=X_train_S.shape[1])
    train_model(model_S, X_train_S, y_train, class_weights=class_weights)
    acc_S = evaluate(model_S, X_test_S, y_test, label="Semantic (test)")

    # --- Train kinematic model ---
    print("\n[4] Training KINEMATIC model...")
    model_K = MLP(input_dim=X_train_K.shape[1])
    train_model(model_K, X_train_K, y_train, class_weights=class_weights)
    acc_K = evaluate(model_K, X_test_K, y_test, label="Kinematic (test, clean)")

    # --- Degradation experiment ---
    print("\n[5] Degradation experiment (kinematic model, test set)...")
    deg_accs = {}
    for level in [0, 1, 2, 3]:
        X_deg = degrade_kinematics(X_test_K, level)
        acc = evaluate(model_K, X_deg, y_test,
                       label=f"Kinematic (test, degradation L{level})")
        deg_accs[level] = acc

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Representation':<16s} | {'Clean':>8s} | {'L1':>8s} | {'L2':>8s} | {'L3':>8s}")
    print("-" * 60)
    print(f"{'Semantic':<16s} | {acc_S*100:>7.1f}% | {'   -':>8s} | {'   -':>8s} | {'   -':>8s}")
    print(f"{'Kinematic':<16s} | {deg_accs[0]*100:>7.1f}% | "
          f"{deg_accs[1]*100:>7.1f}% | {deg_accs[2]*100:>7.1f}% | {deg_accs[3]*100:>7.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
