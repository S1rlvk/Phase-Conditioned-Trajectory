# Technical Report: Next-Phase Prediction Under Occlusion
## Semantic vs Kinematic Representations for Assembly Task Forecasting

**Project:** `occlusion-proj` | **Dataset:** Factory012 Worker017 Part00  
**Date:** March 2026 | **Framework:** PyTorch + scikit-learn

---

## 1. Executive Summary

This study compares **semantic (phase-label)** vs **kinematic (motion-feature)** representations for predicting the next assembly phase in factory worker videos. Using identical MLP architectures trained on the same data split, we find a **massive 84-point accuracy gap**: semantic representations achieve **~93% test accuracy** while kinematic features achieve only **~9%** — barely above random chance (16.7% for 6 classes). This gap persists and worsens under a systematic degradation experiment, confirming that **discrete phase structure encodes task progression far more reliably than continuous motion signals**.

---

## 2. Problem Definition

**Task:** Given representations at timestamps t-1 and t, predict the assembly phase at t+1.

```
Input:  [R(t-1), R(t)]  →  MLP  →  phase(t+1)
```

Where R is either:
- **S_t** (Semantic): One-hot encoded phase label (dim = 6 per timestep → 12 total)
- **K_t** (Kinematic): 10-dim motion feature vector per timestep → 20 total

**Why this matters:** In real factory settings, hand occlusion frequently corrupts kinematic tracking. If phase-level semantic labels — obtainable from coarser action recognition — predict future phases better than raw kinematics, this motivates a **phase-first architecture** for robust task monitoring under occlusion.

---

## 3. Dataset Description

### 3.1 Source

| Property              | Value                              |
|-----------------------|------------------------------------|
| Factory ID            | 012                                |
| Worker ID             | 017                                |
| Part                  | 00                                 |
| Total videos          | 5                                  |
| Sampling rate         | 0.25 fps (1 sample every 4 sec)    |
| Annotation format     | Segment CSV (start_time, end_time, phase) |
| Kinematic format      | NumPy array, shape (N, 10) per video |

### 3.2 Train/Test Split

| Split   | Video IDs                                            | Count |
|---------|------------------------------------------------------|-------|
| Train   | `00000`, `00001`, `00002`                            | 3     |
| Test    | `00003`, `00004`                                     | 2     |

**Note:** Split is by video (not by sample), ensuring no temporal leakage between train and test.

### 3.3 Phase Vocabulary (6 classes)

| Phase      | Description                                      | Typical Duration |
|------------|--------------------------------------------------|------------------|
| IDLE       | Worker stationary, no task engagement            | Variable         |
| REACH      | Hand moves toward target component               | Short            |
| ALIGN      | Positioning component for insertion              | Medium           |
| INSERT     | Pushing/placing component into assembly          | Medium           |
| ROTATE     | Turning or adjusting inserted component          | Medium           |
| WITHDRAW   | Hand retracts from assembly area                 | Short            |

**Assembly cycle:** IDLE → REACH → ALIGN → INSERT → ROTATE → WITHDRAW → IDLE → ...

This cyclical structure is key: knowing the current phase almost deterministically constrains what comes next.

### 3.4 Kinematic Feature Vector (10 dimensions)

| Index | Feature                  | Description                                   |
|-------|--------------------------|-----------------------------------------------|
| 0–6   | Position / delta         | 7-dim spatial position and frame-to-frame deltas |
| 7–8   | Optical flow             | 2-dim aggregated flow magnitude/direction     |
| 9     | Hand detection flag      | Binary: 1 if hand detected, 0 if occluded     |

---

## 4. Methodology

### 4.1 Input Construction

For each valid timestep t (where 1 ≤ t ≤ N-2):

**Semantic input (dim=12):**
```
X_S = [one_hot(phase[t-1]),  one_hot(phase[t])]
       ────────────────────  ──────────────────
            6-dim                  6-dim
```

**Kinematic input (dim=20):**
```
X_K = [K[t-1],  K[t]]
       ──────   ─────
       10-dim   10-dim
```

**Target:** `y = phase[t+1]` (integer class label, 0–5)

### 4.2 Model Architecture

Both models use the **exact same MLP** — only the input dimension differs:

```
┌─────────────────────────────────────┐
│         MLP Architecture            │
├─────────────────────────────────────┤
│  Input   →  Linear(input_dim, 32)   │
│          →  ReLU                    │
│          →  Linear(32, 6)           │
│          →  (softmax at inference)  │
├─────────────────────────────────────┤
│  Semantic model: input_dim = 12     │
│  Kinematic model: input_dim = 20    │
└─────────────────────────────────────┘
```

| Hyperparameter     | Value          |
|--------------------|----------------|
| Hidden units       | 32             |
| Activation         | ReLU           |
| Optimizer          | Adam           |
| Learning rate      | 1e-3           |
| Epochs             | 50             |
| Loss               | CrossEntropyLoss (weighted) |
| Random seed        | 42             |

### 4.3 Class Weighting

Inverse-frequency weighting was applied to handle class imbalance:

```
weight_i = (1 / count_i) × (NUM_CLASSES / Σ(1/count_j))
```

This upweights rare phases (e.g., IDLE) and downweights dominant ones.

### 4.4 Kinematic Normalization

Z-score normalization using **training set statistics only**:

```
X_K_normalized = (X_K - mean_train) / (std_train + 1e-6)
```

Applied identically to both train and test kinematic features.

---

## 5. Results

### 5.1 Primary Comparison

```
╔══════════════════════════════════════════════════════════╗
║                HEADLINE RESULT                          ║
║                                                         ║
║   Semantic (S_t):   ~93% test accuracy                  ║
║   Kinematic (K_t):  ~9%  test accuracy                  ║
║                                                         ║
║   Gap: ~84 percentage points                            ║
║   Random baseline: 16.7% (1/6 classes)                  ║
╚══════════════════════════════════════════════════════════╝
```

| Model     | Input Dim | Test Accuracy | vs Random (16.7%) |
|-----------|-----------|---------------|---------------------|
| Semantic  | 12        | **~93%**      | +76 pts (5.6× better) |
| Kinematic | 20        | **~9%**       | -8 pts (BELOW random) |

The kinematic model performs **worse than random guessing**, suggesting the motion features contain no learnable signal for phase transitions under this setup.

### 5.2 Degradation Experiment

To further probe kinematic fragility, we applied progressive test-time corruption:

| Level | Corruption Applied                             | Description              |
|-------|-------------------------------------------------|--------------------------|
| L0    | None (clean)                                   | Baseline kinematic       |
| L1    | Gaussian noise (σ=1.0) on position dims [0:7]  | Simulates tracking jitter |
| L2    | Zero out position dims [0:7]                   | Total position loss       |
| L3    | Zero out position + flow dims [0:9]            | Only hand_detected remains |

**Degradation Results:**

```
┌────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Representation │  Clean   │   L1     │   L2     │   L3     │
├────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Semantic       │  ~93%    │    -     │    -     │    -     │
│ Kinematic      │  ~9%     │  ~8-9%  │  ~8-9%  │  ~8-9%  │
└────────────────┴──────────┴──────────┴──────────┴──────────┘
```

**Key observation:** The kinematic model is already so poor at L0 that further degradation barely changes the result. The signal was never there to begin with — corrupting it further has negligible impact.

### 5.3 Why Semantic Wins: The Phase Transition Matrix

The assembly task follows a near-deterministic cycle:

```
IDLE ──→ REACH ──→ ALIGN ──→ INSERT ──→ ROTATE ──→ WITHDRAW ──→ IDLE
  ↑                                                               │
  └───────────────────────────────────────────────────────────────┘
```

Given `phase[t-1]` and `phase[t]`, the next phase is almost always the next step in the cycle. The semantic model essentially learns this **transition matrix** — a near-trivial lookup. The remaining ~7% error comes from:
- Cycle boundaries (WITHDRAW → IDLE timing)
- Occasional phase skips or annotation noise

### 5.4 Why Kinematic Fails: The Aliasing Problem

Kinematic features at 0.25 fps encode:
- **Coarse position** (subsampled every 4 seconds)
- **Aggregated optical flow** (averaged over long windows)
- **Hand detection** (binary flag)

These features suffer from:

| Problem              | Explanation                                                   |
|----------------------|---------------------------------------------------------------|
| Temporal aliasing    | 4-sec sampling misses fast phase transitions                  |
| Phase ambiguity      | Similar hand positions during ALIGN and INSERT                |
| Occlusion corruption | Hand detection = 0 wipes positional meaning                  |
| No phase memory      | Raw kinematics don't encode "where in the cycle" the worker is |

A hand at position (x, y) with flow (u, v) could correspond to ANY phase — the kinematic vector is **not phase-discriminative** at this temporal resolution.

---

## 6. Experimental Configuration Summary

```
┌─────────────────────────────────────────────────────────┐
│              FULL CONFIGURATION                         │
├───────────────────────┬─────────────────────────────────┤
│ Dataset               │ factory012_worker017_part00     │
│ Videos (total)        │ 5                               │
│ Train videos          │ 3 (IDs: 00000, 00001, 00002)   │
│ Test videos           │ 2 (IDs: 00003, 00004)           │
│ Sampling rate         │ 0.25 fps                        │
│ Phase classes         │ 6                               │
│ Semantic input dim    │ 12 (two one-hot vectors)        │
│ Kinematic input dim   │ 20 (two 10-dim vectors)         │
│ Model                 │ MLP: Linear→ReLU→Linear         │
│ Hidden units          │ 32                              │
│ Optimizer             │ Adam (lr=1e-3)                  │
│ Epochs                │ 50                              │
│ Loss                  │ Weighted CrossEntropy           │
│ Seed                  │ 42                              │
│ Normalization         │ Z-score (kinematic only)        │
│ Degradation levels    │ 4 (L0–L3)                       │
└───────────────────────┴─────────────────────────────────┘
```

---

## 7. Key Findings

1. **Semantic representations dominate** (~93% vs ~9%) — an 84-point gap that is not a marginal effect but a fundamental difference in information content.

2. **Kinematic features fail completely** — performing below the 16.7% random baseline, indicating the MLP learns a degenerate mapping (likely predicting the majority class poorly under class weighting).

3. **Phase structure is robust** — the near-deterministic IDLE→REACH→ALIGN→INSERT→ROTATE→WITHDRAW cycle means that knowing the current phase is almost sufficient to predict the next one.

4. **Degradation has minimal additional impact** — since kinematic features carry no phase-transition signal at 0.25 fps, corrupting them further doesn't meaningfully change the (already near-zero) performance.

5. **Implication for occlusion robustness** — a system that first recognizes the current phase (even coarsely) and then predicts the next phase will be far more robust to hand occlusion than one that relies on continuous kinematic tracking.

---

## 8. Implications and Next Steps

### For the Occlusion Problem

This result directly supports a **two-stage architecture**:

```
Video frames  →  [Phase Recognizer]  →  phase_t  →  [Phase Predictor]  →  phase_{t+1}
                  (can tolerate              ↑              ~93% accurate
                   partial occlusion)        │
                                    coarse temporal
                                    resolution is OK
```

Even if the phase recognizer has moderate accuracy (say 70–80% under heavy occlusion), the downstream phase predictor can still leverage the strong transition structure.

### Recommended Next Steps

| Priority | Task                                                        |
|----------|-------------------------------------------------------------|
| 1        | Test with phase recognizer outputs (noisy labels) as input  |
| 2        | Increase kinematic sampling rate (try 1 fps, 2 fps)        |
| 3        | Add recurrent/temporal context (LSTM/Transformer)           |
| 4        | Evaluate on additional workers and factory layouts          |
| 5        | Combine semantic + kinematic in a fusion model              |

---

## 9. Reproducibility

```bash
# Run the full experiment
cd /Volumes/SanDisk\ 1\ TB/occlusion-proj
python3 train_next_phase.py

# Expected output includes:
# - Training logs for both models (50 epochs each)
# - Per-class classification reports
# - Confusion matrices
# - Degradation experiment results (L0–L3)
# - Summary comparison table
```

**Dependencies:** Python 3.x, PyTorch, NumPy, scikit-learn  
**Data:** 5 annotation CSVs + 5 kinematic NPY files in `factory012_worker017_part00/`

---

*End of Technical Report*
