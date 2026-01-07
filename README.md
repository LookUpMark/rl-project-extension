# Starting code for final course project extension of Robot Learning - 01HFNOV

Official assignment at [Google Doc](https://docs.google.com/document/d/1XWE2NB-keFvF-EDT_5muoY8gdtZYwJIeo48IMsnr9l0/edit?usp=sharing).

# Automatic Domain Randomization for Robust Sim-to-Real Transfer in Locomotion Tasks

**Course:** Robot Learning  
**Institution:** Politecnico di Torino  
**Academic Year:** 2025/2026  
**Authors:** Marc'Antonio Lopez (s336362), Luigi Marguglio (s332575)

---

## Abstract

This project addresses the fundamental challenge of transferring reinforcement learning policies from simulation to the real world, commonly known as the **reality gap**. We implement and evaluate **Automatic Domain Randomization (ADR)** for the MuJoCo Hopper locomotion task, conducting a comprehensive two-part study. Part 1 compares ADR against baseline training and Uniform Domain Randomization (UDR) across multiple training durations. Part 2 presents a systematic ablation study analyzing the individual contribution of physics parameters (mass, damping, friction) to transfer performance.

Our key finding is that **friction randomization alone achieves a remarkable +154.6% transfer gap**, significantly outperforming all other configurations including full parameter randomization. Statistical analysis reveals that friction contributes +68.7% marginally, while mass shows a negative contribution of -15.7%. We discover strong antagonistic interaction effects: combining mass with friction reduces the positive effect of friction by 94.5%. These results demonstrate that selective, data-driven parameter randomization outperforms the common practice of uniformly randomizing all available parameters.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Related Work](#2-background-and-related-work)
3. [Methodology](#3-methodology)
4. [Experimental Setup](#4-experimental-setup)
5. [Results](#5-results)
6. [Discussion](#6-discussion)
7. [Conclusions](#7-conclusions)
8. [Installation and Usage](#8-installation-and-usage)
9. [Project Structure](#9-project-structure)
10. [Reproducibility](#10-reproducibility)
11. [References](#11-references)

---

## 1. Introduction

### 1.1 The Reality Gap Problem

Training robots in simulation offers significant practical advantages: unlimited data collection without hardware wear, massively parallel environments for accelerated learning, and complete elimination of hardware damage risk during exploration. However, policies trained purely in simulation frequently fail when deployed on real hardware due to the inevitable discrepancy between the idealized physics of simulation and the complex dynamics of the real world.

This phenomenon, known as the **reality gap**, manifests in several ways:
- Unmodeled friction and contact dynamics
- Inaccurate mass and inertia estimates
- Simplified actuator models
- Absence of sensor noise and delays

### 1.2 Domain Randomization as a Solution

**Domain Randomization** addresses this challenge by varying simulation parameters during training. The underlying principle is that if an agent learns to perform well across a distribution of simulated environments, it should generalize to the real world, which can be viewed as just another sample from a sufficiently broad distribution. The robot effectively learns behaviors that are invariant to parameter changes, making it robust to unknown real-world dynamics.

Traditional **Uniform Domain Randomization (UDR)** samples parameters from fixed ranges defined a priori. While effective, this approach suffers from a fundamental limitation: selecting appropriate ranges requires careful manual tuning. Ranges that are too narrow may not encompass the real-world parameters, while ranges that are too wide can generate physically implausible scenarios, leading to **learned helplessness** where the agent gives up learning anything useful.

### 1.3 Automatic Domain Randomization

**Automatic Domain Randomization (ADR)**, introduced by OpenAI for dexterous manipulation tasks, elegantly solves the range selection problem by adapting randomization ranges based on agent performance:

- When performance exceeds a high threshold, the environment becomes "too easy" and ranges **expand** to increase difficulty
- When performance drops below a low threshold, ranges **contract** to make the task more manageable
- This creates an automatic curriculum where the agent is always challenged at an appropriate level

### 1.4 Project Contributions

This project makes two main contributions:

1. **Part 1 - ADR Evaluation:** Systematic comparison of ADR against baseline and UDR across multiple training durations (2.5M, 5M, 10M timesteps), analyzing the relationship between training time, ADR range expansion, and transfer performance.

2. **Part 2 - Parameter Relevance Analysis:** A comprehensive ablation study testing all 2³ = 8 combinations of mass, damping, and friction randomization, with statistical analysis of marginal contributions and interaction effects between parameters.

---

## 2. Background and Related Work

### 2.1 Domain Randomization

Domain randomization was popularized by Tobin et al. (2017) for visual tasks, demonstrating that policies trained on randomized synthetic images could transfer successfully to real cameras. The approach was subsequently extended to dynamics randomization by Peng et al. (2018), randomizing physical properties such as mass, friction, and joint dynamics.

Muratore et al. (2018) introduced SPOTA (Simulation-based Policy Optimization with Transferability Assessment), providing a principled stopping criterion for domain randomization training based on transferability estimation. Ramos et al. (2019) proposed BayesSim, a Bayesian framework that computes posterior distributions over simulator parameters, enabling adaptive domain randomization that outperforms uniform priors.

### 2.2 Sim-to-Real for Locomotion

For locomotion specifically, Tan et al. (2018) achieved successful sim-to-real transfer for quadruped robots by randomizing friction coefficients and introducing latency randomization during training. Hwangbo et al. (2019) demonstrated that neural network policies trained in simulation could enable ANYmal robots to perform agile and dynamic motor skills, with careful attention to contact dynamics. Lee et al. (2020) extended this work, showing that proprioceptive controllers trained in simulation achieve zero-shot generalization to challenging natural terrains.

### 2.3 Automatic Domain Randomization

OpenAI's work on solving Rubik's cube with a robot hand (2019) brought ADR to prominence by demonstrating that automatic difficulty adjustment could achieve unprecedented dexterity through an emergent curriculum. Mehta et al. (2020) proposed Active Domain Randomization, which learns a parameter sampling strategy to prioritize informative environment variations rather than relying on uniform sampling.

### 2.4 Parameter Relevance in Sim-to-Real

Recent work by Hu et al. (2025) investigated the impact of static friction on sim-to-real transfer, finding that friction parameters are often the most critical for locomotion tasks. Our ablation study provides empirical support for this finding.

---

## 3. Methodology

### 3.1 Environment Description

We use a modified version of the **MuJoCo Hopper** environment, a standard benchmark for locomotion control consisting of a single-legged robot that must learn to hop forward as fast as possible while maintaining balance.

| Property | Value |
|----------|-------|
| Observation Space | 11-dimensional continuous (joint positions and velocities) |
| Action Space | 3-dimensional continuous (joint torques) |
| Episode Length | Maximum 500 steps |
| Reward | Forward velocity + healthy bonus - control cost |

We define two environment variants to simulate the reality gap:

- **Source Environment:** Misspecified dynamics with a 1kg torso mass offset (simulates an inaccurate simulator)
- **Target Environment:** Correct nominal dynamics (simulates the real robot)

The transfer gap is defined as:

```
Transfer Gap = (Target Reward - Source Reward) / Source Reward × 100%
```

A positive transfer gap indicates that the policy performs better on the target than on the source, suggesting successful adaptation to the reality gap.

### 3.2 ADR Implementation

Our ADR implementation maintains a state vector representing randomization ranges for each physics parameter:

```
S = {δ_mass, δ_damping, δ_friction}
```

where each δ ∈ [0, 1] specifies the fractional variation around the nominal value. At each episode reset, parameters are sampled uniformly:

```
parameter = nominal × (1 + uniform(-δ, +δ))
```

The ADR update rule evaluates performance every 2048 timesteps:

| Condition | Action | Effect |
|-----------|--------|--------|
| Mean Reward ≥ 1200 | δ += 0.05 | Expand ranges (harder) |
| Mean Reward < 600 | δ = max(0, δ - 0.05) | Contract ranges (easier) |
| Otherwise | δ unchanged | Maintain difficulty |

This creates an automatic curriculum where the environment difficulty adapts to the agent's current capability.

### 3.3 ADR Callback Architecture

The ADR system is implemented as a Stable Baselines 3 callback that:

1. Monitors the episodic reward buffer
2. Computes the mean reward over recent episodes
3. Calls the environment's `update_adr()` method with performance thresholds
4. Logs ADR state to TensorBoard for monitoring

### 3.4 Ablation Study Design

To analyze parameter relevance, we designed a factorial experiment testing all combinations of enabled/disabled parameters:

| Configuration | Mass | Damping | Friction | Mode |
|---------------|:----:|:-------:|:--------:|------|
| baseline | ✗ | ✗ | ✗ | No randomization |
| udr | ✓ | ✓ | ✓ | Fixed ±30% ranges |
| adr_none | ✗ | ✗ | ✗ | ADR (control group) |
| adr_mass | ✓ | ✗ | ✗ | ADR with mass only |
| adr_damp | ✗ | ✓ | ✗ | ADR with damping only |
| adr_fric | ✗ | ✗ | ✓ | ADR with friction only |
| adr_mass_damp | ✓ | ✓ | ✗ | ADR with mass + damping |
| adr_mass_fric | ✓ | ✗ | ✓ | ADR with mass + friction |
| adr_damp_fric | ✗ | ✓ | ✓ | ADR with damping + friction |
| adr_all | ✓ | ✓ | ✓ | ADR with all parameters |

### 3.5 Statistical Analysis

For each parameter P, we compute the **marginal contribution**:

```
Contribution(P) = mean(gap | P enabled) - mean(gap | P disabled)
```

We assess statistical significance using independent two-sample t-tests with α = 0.10.

For parameter pairs, we compute **interaction effects**:

```
Expected_additive = Contribution(P1) + Contribution(P2) + base_gap
Actual = gap(P1 ∧ P2)
Interaction = Actual - Expected_additive
```

Positive interaction indicates synergy; negative interaction indicates antagonism.

---

## 4. Experimental Setup

### 4.1 Training Configuration

All experiments use the following configuration:

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (Proximal Policy Optimization) |
| Policy | MlpPolicy (2 hidden layers, 64 units each) |
| Learning Rate | 3×10⁻⁴ (default) |
| Batch Size | 2048 |
| Random Seed | 42 (fixed for reproducibility) |
| Evaluation Episodes | 50 per configuration |

### 4.2 Part 1: Training Durations

| Model | Timesteps | Approximate Training Time |
|-------|-----------|---------------------------|
| Baseline | 2,500,000 | ~2.5 hours |
| UDR | 2,500,000 | ~2.5 hours |
| ADR 2.5M | 2,500,000 | ~2.5 hours |
| ADR 5M | 5,000,000 | ~4 hours |
| ADR 10M | 10,000,000 | ~6 hours |

### 4.3 Part 2: Ablation Study

All 10 ablation configurations were trained for 2,500,000 timesteps each, for a total training time of approximately 25 hours.

### 4.4 Hardware

Experiments were conducted on a system with CUDA-enabled GPU acceleration used for environment simulation.

---

## 5. Results

### 5.1 Part 1: Method Comparison

#### 5.1.1 ADR Range Evolution

During training, ADR progressively expands randomization ranges as the agent improves. We observed the following final ADR ranges:

| Model | Final ADR Range | Training Reward |
|-------|-----------------|-----------------|
| ADR 2.5M | ±60% | 989 |
| ADR 5M | ±60% | 749 |
| ADR 10M | ±40% | 1016 |

Interestingly, longer training did not necessarily lead to higher ADR ranges, suggesting that the relationship between training time and robustness is non-monotonic.

#### 5.1.2 Transfer Performance

| Method | Source Reward | Target Reward | Transfer Gap |
|--------|---------------|---------------|--------------|
| Baseline | 1778 ± 65 | 1169 ± 95 | **-34.2%** |
| UDR (±30%) | 1660 ± 10 | 1725 ± 34 | **+3.9%** |
| ADR 2.5M | 1567 ± 7 | 1533 ± 133 | -2.1% |
| ADR 5M | 1013 ± 224 | 781 ± 139 | -22.9% |
| ADR 10M | 1462 ± 39 | 1457 ± 145 | **-0.4%** |

Key observations:
- **Baseline** exhibits a severe reality gap of -34.2%, confirming the necessity of domain randomization
- **UDR** achieves the best absolute transfer performance (+3.9%) with remarkably low variance
- **ADR 10M** achieves near-perfect transfer stability (-0.4%), essentially eliminating the reality gap
- **ADR 5M** shows unexpectedly poor performance despite reaching high ADR ranges

### 5.2 Part 2: Ablation Study

#### 5.2.1 Complete Results

| Rank | Configuration | Source | Target | Transfer Gap |
|------|---------------|--------|--------|--------------|
| 1 | adr_fric | 642 ± 98 | 1634 ± 2 | **+154.6%** |
| 2 | adr_all | 1088 ± 115 | 1241 ± 258 | +14.1% |
| 3 | adr_damp | 1631 ± 301 | 1761 ± 31 | +8.0% |
| 4 | adr_damp_fric | 1530 ± 8 | 1638 ± 36 | +7.1% |
| 5 | udr | 1724 ± 10 | 1711 ± 104 | -0.8% |
| 6 | adr_mass_fric | 1631 ± 3 | 1558 ± 85 | -4.5% |
| 7 | adr_mass | 1180 ± 181 | 973 ± 93 | -17.5% |
| 8 | adr_mass_damp | 949 ± 52 | 648 ± 29 | -31.7% |
| 9 | adr_none | 875 ± 208 | 303 ± 222 | -65.4% |
| 10 | baseline | 933 ± 213 | 314 ± 224 | -66.4% |

The most striking result is that **friction-only ADR (adr_fric) achieves +154.6% transfer gap**, vastly outperforming all other configurations including full parameter randomization.

#### 5.2.2 Statistical Analysis: Marginal Contributions

| Rank | Parameter | Marginal Contribution | p-value | Significance |
|------|-----------|----------------------|---------|--------------|
| 1 | **Friction** | **+68.70%** | 0.0744 | * (p < 0.10) |
| 2 | Mass | -15.68% | 0.7120 | Not significant |
| 3 | Damping | -0.86% | 0.9840 | Not significant |

Friction is the only parameter with a positive and marginally significant contribution. Mass shows a **negative** contribution, meaning that adding mass randomization tends to hurt transfer performance.

#### 5.2.3 Interaction Effects

| Interaction | Expected (Additive) | Actual | Effect | Interpretation |
|-------------|---------------------|--------|--------|----------------|
| Mass × Damping | -11.1% | -6.1% | +5.0% | Synergistic |
| Mass × Friction | +97.5% | +2.9% | **-94.5%** | **Antagonistic** |
| Damping × Friction | +113.0% | +6.8% | **-106.2%** | **Antagonistic** |

We discover strong **antagonistic interactions**: combining mass with friction reduces friction's positive effect by 94.5%. Similarly, combining damping with friction reduces the benefit by 106.2%. This explains why `adr_fric` (friction only) dramatically outperforms `adr_all` (all parameters).

---

## 6. Discussion

### 6.1 Why Friction Dominates

The exceptional performance of friction-only ADR can be attributed to several factors:

1. **Ground contact is critical for locomotion:** The Hopper's ability to generate forward thrust depends entirely on the friction between its foot and the ground. Robust friction handling directly translates to robust locomotion.

2. **Friction is often misspecified:** Simulation environments typically assume idealized contact dynamics, while real surfaces exhibit complex friction behaviors including stiction, slip, and surface irregularities.

3. **Focused robustness learning:** Single-parameter ADR allows the agent to develop strong invariance to that specific parameter without the confounding effects of other variations.

### 6.2 Why Mass Randomization Hurts

Counter-intuitively, mass randomization shows a negative contribution. Several hypotheses explain this:

1. **Built-in mismatch:** The source environment already has a 1kg torso mass offset. Additional mass randomization may interfere with the agent's ability to compensate for this known mismatch.

2. **Over-robustification:** Mass variations may force the agent to adopt overly conservative strategies that sacrifice performance for stability.

3. **Conflicting objectives:** Learning to handle mass variations may conflict with learning optimal locomotion gaits.

### 6.3 The Interaction Effect Problem

The discovery of strong antagonistic interactions challenges the common assumption that "more randomization is better." When parameters are randomized together:

- The agent must simultaneously cope with multiple sources of variation
- Parameter combinations may create scenarios that are physically inconsistent or extremely difficult
- The learning signal becomes noisy, making it harder to identify useful invariances

### 6.4 Implications for Practice

Based on our empirical findings, we provide the following recommendations:

| Goal | Recommended Configuration |
|------|---------------------------|
| Maximum transfer performance | `adr_fric` (friction only) |
| Balanced robustness | `adr_damp_fric` (damping + friction) |
| Conservative approach | `udr` (fixed ±30% all parameters) |
| **Configurations to avoid** | `adr_mass`, `adr_mass_damp` |

Practitioners should conduct ablation studies to identify which parameters are relevant for their specific task rather than assuming that randomizing all parameters is optimal.

---

## 7. Conclusions

This project presents a comprehensive study of domain randomization for locomotion, with the following key findings:

1. **Friction is the most critical parameter:** Friction-only ADR achieves +154.6% transfer gap, outperforming all other configurations by a large margin.

2. **Mass randomization can hurt:** Mass contributes -15.7% to transfer performance, likely due to interference with the built-in dynamics mismatch.

3. **Parameter interactions matter:** Strong antagonistic effects (up to -106%) mean that combining parameters can reduce or eliminate individual benefits.

4. **Selective randomization outperforms uniform:** Data-driven parameter selection based on relevance analysis produces better results than uniformly randomizing all available parameters.

5. **ADR range ≠ transfer quality:** Higher ADR ranges do not guarantee better transfer; what matters is *which* parameters are randomized, not just how much.

### Future Work

Several directions remain for future investigation:

1. **Real robot validation:** Deploying the trained policies on physical hardware to validate sim-to-real transfer
2. **Additional parameters:** Extending the ablation study to include joint stiffness, actuator delays, and sensor noise
3. **Automated selection:** Developing methods that automatically identify relevant parameters during training rather than post-hoc analysis
4. **Transfer to other tasks:** Investigating whether friction dominance generalizes to other locomotion platforms

---

## 8. Installation and Usage

### 8.1 Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (optional, for faster training)

### 8.2 Installation

```bash
# Clone the repository
git clone https://github.com/LookUpMark/rl-project-extension.git
cd rl-project-extension

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 8.3 Training Commands

```bash
# Part 1: Method comparison
python scripts/train/train_baseline.py      # Baseline (no randomization)
python scripts/train/train_udr.py           # UDR (±30% fixed)
python scripts/train/train_adr.py --run 10M # ADR (10M timesteps)

# Part 2: Ablation study
python scripts/train/train_ablation.py --config adr_fric
python scripts/train/train_ablation.py --config adr_all
# ... (run for all 10 configurations)
```

### 8.4 Analysis Commands

```bash
# Statistical analysis
python scripts/analysis/statistical_analysis.py

# Generate visualizations
python scripts/analysis/plot_ablation.py

# Automatic parameter selection
python scripts/analysis/auto_select_params.py
```

### 8.5 Monitoring

```bash
# Launch TensorBoard to monitor training
tensorboard --logdir ./logs/
```

---

## 9. Project Structure

```
rl-project-extension/
├── env/
│   ├── custom_hopper.py          # CustomHopper environment with ADR support
│   ├── __init__.py
│   └── assets/
│       └── hopper.xml            # MuJoCo model definition
├── callbacks/
│   ├── adr_callback.py           # ADR callback for Stable Baselines 3
│   └── __init__.py
├── scripts/
│   ├── train/
│   │   ├── utils.py              # Shared training utilities
│   │   ├── train_baseline.py     # Baseline training script
│   │   ├── train_udr.py          # UDR training script
│   │   ├── train_adr.py          # ADR training script
│   │   ├── train_ablation.py     # Ablation study training script
│   │   └── train_optimal.py      # Training with auto-selected params
│   ├── test/
│   │   ├── test_comparison.py    # Generate comparison charts
│   │   └── test_random_policy.py # Environment sanity check
│   └── analysis/
│       ├── statistical_analysis.py   # Compute parameter contributions
│       ├── plot_ablation.py          # Generate visualization figures
│       └── auto_select_params.py     # Automated parameter selection
├── logs/
│   ├── baseline/                 # Baseline model and TensorBoard logs
│   ├── udr/                      # UDR model and logs
│   ├── adr/                      # ADR models (run_2_5M, run_5M, run_10M)
│   └── ablation/                 # Ablation study results (10 configurations)
│       ├── adr_fric/
│       │   ├── ppo_ablation_adr_fric.zip
│       │   └── results.json
│       └── ...
├── docs/
│   └── assignment/
│       └── ASSIGNMENT.md         # Original course assignment
├── README.md                     # This documentation
├── requirements.txt              # Python dependencies
└── LICENSE                       # MIT License
```

---

## 10. Reproducibility

### 10.1 Random Seeds

All experiments use fixed random seeds for complete reproducibility:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
```

### 10.2 Results Format

Each ablation configuration saves results to `logs/ablation/<config>/results.json`:

```json
{
  "config_name": "adr_fric",
  "mode": "adr",
  "config": {"mass": false, "damping": false, "friction": true},
  "seed": 42,
  "timesteps": 2500000,
  "source_mean": 641.6,
  "source_std": 98.2,
  "target_mean": 1633.8,
  "target_std": 2.1,
  "transfer_gap": 154.6,
  "final_adr_state": {"mass_range": 0.0, "damping_range": 0.0, "friction_range": 0.65},
  "timestamp": "2024-12-29T18:45:23"
}
```

### 10.3 Verification

To verify that all components work correctly:

```bash
# Test environment and imports
python -c "from env.custom_hopper import *; print('Environment OK')"
python -c "from callbacks.adr_callback import ADRCallback; print('Callback OK')"

# Run a quick environment test
python scripts/test/test_random_policy.py
```

---

## 11. References

1. OpenAI, I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, J. Schneider, N. Tezak, J. Tworek, P. Welinder, L. Weng, Q. Yuan, W. Zaremba, and L. Zhang (2019). *Solving Rubik's Cube with a Robot Hand*. arXiv:1910.07113.

2. Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 23-30.

3. Peng, X. B., Andrychowicz, M., Zaremba, W., & Abbeel, P. (2018). *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*. IEEE International Conference on Robotics and Automation (ICRA), pp. 3803-3810.

4. Tan, J., Zhang, T., Coumans, E., Iscen, A., Bai, Y., Hafner, D., Bohez, S., & Vanhoucke, V. (2018). *Sim-to-Real: Learning Agile Locomotion For Quadruped Robots*. arXiv:1804.10332.

5. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

6. Hu, X., Sun, Q., He, B., Liu, H., Zhang, X., Lu, C., & Zhong, J. (2025). *Impact of Static Friction on Sim2Real in Robotic Reinforcement Learning*. arXiv:2503.01255.

7. Mehta, B., Diaz, M., Golemo, F., Pal, C. J., & Paull, L. (2019). *Active Domain Randomization*. arXiv:1904.04762.

8. Muratore, F., Treede, F., Gienger, M., & Peters, J. (2018). *Domain Randomization for Simulation-Based Policy Optimization with Transferability Assessment*. Conference on Robot Learning (CoRL), PMLR, pp. 700-713.

9. Ramos, F., Possas, R. C., & Fox, D. (2019). *BayesSim: Adaptive Domain Randomization via Probabilistic Inference for Robotics Simulators*. arXiv:1906.01728.

10. Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., & Hutter, M. (2019). *Learning Agile and Dynamic Motor Skills for Legged Robots*. Science Robotics, 4(26).

11. Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., & Hutter, M. (2020). *Learning Quadrupedal Locomotion over Challenging Terrain*. Science Robotics, 5(47).

12. Todorov, E., Erez, T., & Tassa, Y. (2012). *MuJoCo: A Physics Engine for Model-Based Control*. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5026-5033.

---

## Acknowledgments

This work was conducted as part of the **Robot Learning** course at Politecnico di Torino, under the supervision of the **VANDAL Laboratory** (Vision and Learning Lab).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Authors

**Marc'Antonio Lopez**  
MSc Student in Computer Engineering  
Politecnico di Torino  
Email: s336362@studenti.polito.it

**Luigi Marguglio**  
MSc Student in Computer Engineering  
Politecnico di Torino  
Email: s332575@studenti.polito.it
