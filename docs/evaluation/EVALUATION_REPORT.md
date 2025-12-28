# Automatic Domain Randomization (ADR) Evaluation Report

**Project:** Sim-to-Real Hopper with ADR Extension  
**Date:** December 28, 2024  
**Seed:** 42 (fixed for reproducibility)  
**Models:** `logs/baseline/`, `logs/udr/`, `logs/adr/`

---

## Executive Summary

This report presents the final results of training PPO agents on the MuJoCo Hopper environment using different domain randomization strategies. All experiments use a fixed random seed (42) for reproducibility.

### Key Results

| Method | Source | Target | Transfer Gap | ADR Range |
|--------|--------|--------|--------------|-----------|
| **Baseline** | 1778 ± 65 | 1169 ± 95 | **-34.2%** | N/A |
| **UDR** | 1660 ± 10 | 1725 ± 34 | **+3.9%** | ±30% fixed |
| **ADR 2.5M** | 1567 ± 7 | 1533 ± 133 | -2.1% | ±60% |
| **ADR 5M** | 1013 ± 224 | 781 ± 139 | -22.9% | ±60% |
| **ADR 10M** | 1462 ± 39 | 1457 ± 145 | **-0.4%** | ±40% |

---

## 1. Training Configuration

### 1.1 Environment Setup

```
Environment: CustomHopper-source-v0 / CustomHopper-target-v0
Observation Space: Box(-inf, inf, (11,), float64)
Action Space: Box(-1.0, 1.0, (3,), float32)
Source Dynamics: Torso mass -1kg offset (simulates misspecified model)
Target Dynamics: Correct nominal values
```

### 1.2 ADR Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `threshold_high` | 1200 | Reward threshold for expansion |
| `threshold_low` | 600 | Reward threshold for contraction |
| `adr_step_size` | 0.05 | Increment per expansion (5%) |
| `check_freq` | 2048 | Steps between ADR checks |

### 1.3 Reproducibility

All experiments use:
- **Seed:** 42
- **NumPy seed:** 42
- **PyTorch seed:** 42
- **CUDA deterministic:** True

---

## 2. Training Summary

### 2.1 Models Trained

| Model | Timesteps | Training Time | Log Directory |
|-------|-----------|---------------|---------------|
| Baseline | 2,500,000 | ~2h 38m | `logs/baseline/PPO_1/` |
| UDR | 2,500,000 | ~2h 38m | `logs/udr/PPO_1/` |
| ADR 2.5M | 2,500,000 | ~2h 38m | `logs/adr/run_2_5M/PPO_1/` |
| ADR 5M | 5,000,000 | ~4h 03m | `logs/adr/run_5M/PPO_1/` |
| ADR 10M | 10,000,000 | ~6h 15m | `logs/adr/run_10M/PPO_1/` |

### 2.2 ADR Range Evolution

| Model | Final ADR Range | Final Reward (training) |
|-------|-----------------|-------------------------|
| ADR 2.5M | ±60% | 989 |
| ADR 5M | ±60% | 749 |
| ADR 10M | ±40% | 1016 |

**Note:** ADR range is not monotonic with training duration. The stochastic nature of expansion depends on the agent's learning trajectory.

---

## 3. Evaluation Results

### 3.1 Standard Evaluation (Clean Source)

All models evaluated on 50 episodes with `udr=False` for fair comparison.

| Method | Source | Target | Gap |
|--------|--------|--------|-----|
| Baseline | 1778 ± 65 | 1169 ± 95 | -34.2% |
| UDR | 1660 ± 10 | **1725 ± 34** | **+3.9%** |
| ADR 2.5M | 1567 ± 7 | 1533 ± 133 | -2.1% |
| ADR 5M | 1013 ± 224 | 781 ± 139 | -22.9% |
| ADR 10M | 1462 ± 39 | 1457 ± 145 | -0.4% |

### 3.2 Robustness Evaluation (Randomized Source)

ADR models evaluated with `udr=True` on source to simulate training conditions.

| Model | Source (UDR) | Target | Gap |
|-------|--------------|--------|-----|
| ADR 2.5M | 1542 ± 125 | 1541 ± 124 | -0.1% |
| ADR 5M | 804 ± 167 | 743 ± 110 | -7.6% |
| ADR 10M | 1488 ± 149 | **1533 ± 126** | **+3.0%** |

---

## 4. Analysis

### 4.1 Baseline: Reality Gap Confirmed

- **-34.2% transfer gap** demonstrates the classic sim-to-real problem
- Policy overfits to training dynamics and fails on target
- This confirms the necessity of domain randomization

### 4.2 UDR: Best Transfer Performance

- **+3.9% positive transfer** with remarkably low variance (±10 on source)
- The fixed ±30% range is well-calibrated for this task
- Simple and effective when ranges are known a priori

### 4.3 ADR 10M: Best Transfer Stability

- **-0.4% gap** is essentially zero transfer degradation
- Low variance (±39 on source) indicates consistent behavior
- Best choice when deployment stability is critical

### 4.4 ADR 5M: Unexpected Poor Performance

- Despite reaching ±60% ADR range, shows -22.9% gap
- Possible over-robustification: learned overly conservative strategies
- High variance indicates unstable behavior on fixed dynamics

---

## 5. Conclusions

### 5.1 Key Findings

1. **UDR wins on transfer performance** when ranges are well-calibrated
2. **ADR 10M wins on stability** with near-zero transfer gap
3. **Higher ADR range ≠ better transfer** (ADR 5M with 60% underperforms)
4. **Baseline confirms necessity** of domain randomization (-34.2% gap)

### 5.2 Recommendations

| Use Case | Recommended Method |
|----------|-------------------|
| Maximum transfer performance | UDR with tuned ranges |
| Maximum transfer stability | ADR 10M |
| Unknown target dynamics | ADR (any duration) |
| Quick experiments | Baseline (but expect gap) |

### 5.3 Future Work

1. Investigate why ADR 5M underperforms despite high range
2. Develop early stopping criteria based on transfer estimation
3. Apply to real robot hardware for validation

---

## 6. Artifacts

### 6.1 Saved Models

| Model | Path |
|-------|------|
| Baseline | `logs/baseline/ppo_hopper_baseline.zip` |
| UDR | `logs/udr/ppo_hopper_udr.zip` |
| ADR 2.5M | `logs/adr/ppo_hopper_adr_2_5M.zip` |
| ADR 5M | `logs/adr/ppo_hopper_adr_5M.zip` |
| ADR 10M | `logs/adr/ppo_hopper_adr_10M.zip` |

### 6.2 Figures

| Figure | Path |
|--------|------|
| Training Curves | `docs/evaluation/figures/training_curves.png` |
| Learning Curves | `docs/evaluation/figures/learning_curves.png` |
| Final Performance | `docs/evaluation/figures/final_performance.png` |
| Robustness | `docs/evaluation/figures/robustness_performance.png` |

### 6.3 LaTeX Table

```latex
\input{docs/evaluation/figures/results_table.tex}
```

---

## 7. Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (with seed=42)
python scripts/train/train_baseline.py
python scripts/train/train_udr.py
python scripts/train/train_adr.py --run 2_5M
python scripts/train/train_adr.py --run 5M
python scripts/train/train_adr.py --run 10M

# Generate comparison charts
python scripts/test/test_comparison.py

# Monitor with Tensorboard
tensorboard --logdir ./logs/
```

---

**Report Generated:** December 28, 2024  
**Total Training Time:** ~18 hours  
**Hardware:** CUDA-enabled GPU  
**Seed:** 42 (reproducible)
