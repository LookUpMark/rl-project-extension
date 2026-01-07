# Complete Implementation Guide: Sim-to-Real Hopper with ADR

**Version:** 5.0 (Team Edition - 2 Members)  
**Status:** Final for Exam

---

## Team Structure

| Role | Main Responsibility | Sections |
|------|---------------------|----------|
| **Member 1** | Core RL & Environment | CustomHopper, ADR Callback, Debug |
| **Member 2** | Integration & DevOps | Setup, Training Loop, Execution, Analysis |

---

## 1. Concept: Why ADR?

### 1.1 The Problem: Sim-to-Real Paradox

Training robots in simulation (Sim) to act in the real world (Real) is difficult due to the **Reality Gap**.

- **Naive Approach:** Train on a fixed model → Catastrophic failure in reality (overfitting)
- **UDR Approach:** Randomize parameters within fixed ranges
  - *Narrow ranges:* May not cover reality
  - *Wide ranges:* Generates impossible scenarios → *Learned Helplessness*

### 1.2 The Solution: Automatic Domain Randomization (ADR)

ADR transforms training into an **automatic curriculum**:

1. Start from a deterministic environment (easy)
2. If agent performs well (high reward) → environment becomes harder (increase randomization)
3. If agent fails (low reward) → environment becomes easier (decrease randomization)

**Goal:** Obtain a policy that survives maximum entropy (chaos) = Robustness.

---

## 2. Environment: CustomHopper (`env/custom_hopper.py`)

The environment extends the base Hopper with ADR support:

### Key Components

```python
# ADR State - tracks current randomization ranges
self.adr_state = {
    "mass_range": 0.0,      # 0 = deterministic
    "damping_range": 0.0,
    "friction_range": 0.0
}

# Methods
def sample_parameters(self):   # Samples physics params based on ADR state
def set_parameters(self, p):   # Applies sampled params to MuJoCo model
def update_adr(self, r, lo, hi): # Expands/contracts ranges
def get_adr_info(self):        # Returns current ADR state for logging
```

### ADR Update Logic

```python
if mean_reward >= threshold_high:   # 1200
    adr_state[k] += 0.05            # Expand
elif mean_reward < threshold_low:   # 600
    adr_state[k] = max(0, adr_state[k] - 0.05)  # Contract
```

---

## 3. ADR Callback (`callbacks/adr_callback.py`)

Monitors performance and adjusts environment difficulty:

```python
class ADRCallback(BaseCallback):
    def __init__(self, check_freq=2048, threshold_high=1200, threshold_low=600):
        ...
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            env = self.training_env.envs[0].unwrapped
            status, adr = env.update_adr(mean_reward, self.threshold_low, self.threshold_high)
            # Log to tensorboard
            self.logger.record("adr/mass_range", adr["mass_range"])
            ...
```

---

## 4. Training Scripts (`scripts/train/`)

### 4.1 Shared Utilities (`utils.py`)

```python
def set_seed(seed=42):     # Set all random seeds
def create_envs():         # Create source and target environments
def train_and_evaluate():  # Train PPO and evaluate on both envs
```

### 4.2 Available Scripts

| Script | Purpose | Command |
|--------|---------|---------|
| `train_baseline.py` | No randomization | `python scripts/train/train_baseline.py` |
| `train_udr.py` | Fixed ±30% UDR | `python scripts/train/train_udr.py` |
| `train_adr.py` | ADR (2.5M/5M/10M) | `python scripts/train/train_adr.py --run 10M` |
| `train_ablation.py` | Ablation study | `python scripts/train/train_ablation.py --config adr_fric` |
| `train_optimal.py` | Auto-selected params | `python scripts/train/train_optimal.py` |

---

## 5. Execution & Monitoring

### 5.1 Running Training

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train ADR
python scripts/train/train_adr.py --run 10M
```

### 5.2 Monitor with Tensorboard

```bash
tensorboard --logdir ./logs/
```

Key metrics:
- `adr/mean_reward` - Agent performance
- `adr/mass_range` - Mass randomization range
- `adr/friction_range` - Friction randomization range

---

## 6. Analysis (`scripts/analysis/`)

### Statistical Analysis

```bash
python scripts/analysis/statistical_analysis.py
```

Output: `logs/ablation/analysis_report.json` with:
- Marginal contributions per parameter
- Interaction effects
- Recommended parameters

### Visualization

```bash
python scripts/analysis/plot_ablation.py
```

Generates figures in `docs/evaluation/figures/`.

---

## 7. Project Verification

```bash
# Test imports
python -c "from env.custom_hopper import *; from callbacks.adr_callback import ADRCallback"

# Test environment
python scripts/test/test_random_policy.py
```

---

## 8. Key Results

### Part 1: Method Comparison

| Method | Target Reward | Transfer Gap |
|--------|---------------|--------------|
| Baseline | 1169±95 | -34.2% |
| UDR | **1725±34** | **+3.9%** |
| ADR 10M | 1457±145 | -0.4% |

### Part 2: Ablation Study

**Best Config:** `adr_fric` (friction only) → **+154.6%** transfer gap

**Parameter Contributions:**
- FRICTION: +68.7% (most important)
- DAMPING: -0.9%
- MASS: -15.7% (can hurt!)

---

## References

1. OpenAI et al., "Solving Rubik's Cube with a Robot Hand", 2019
2. Mehta et al., "Active Domain Randomization", CoRL 2020
3. Tan et al., "Sim-to-Real: Learning Agile Locomotion", RSS 2018
4. Gang et al., "Impact of Static Friction on Sim2Real", 2025