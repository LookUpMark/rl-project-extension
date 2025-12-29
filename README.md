# Sim-to-Real Hopper with Automatic Domain Randomization (ADR)

An advanced reinforcement learning project implementing **Automatic Domain Randomization** for robust Sim-to-Real transfer in the MuJoCo Hopper environment.

## Overview

This project extends the standard Hopper-v4 environment to implement ADR (Automatic Domain Randomization), a curriculum-based technique that automatically adjusts the difficulty of domain randomization based on agent performance. This produces policies that are robust to the reality gap without manual hyperparameter tuning.

### Key Results (Seed=42)

| Method | Target Reward | Transfer Gap |
|--------|---------------|--------------|
| **Baseline** | 1169 ± 95 | -34.2% |
| **UDR (±30%)** | **1725 ± 34** | **+3.9%** |
| **ADR 10M (±40%)** | 1457 ± 145 | **-0.4%** |

- **UDR** achieves best transfer performance when ranges are well-calibrated
- **ADR 10M** achieves best transfer stability (near-zero gap)
- **Baseline** confirms the reality gap problem (-34.2%)

## Project Structure

```
├── env/
│   ├── custom_hopper.py     # Extended Hopper with ADR + ablation support
│   └── assets/hopper.xml
├── callbacks/
│   └── adr_callback.py      # ADR training callback
├── scripts/
│   ├── train/
│   │   ├── train_baseline.py   # Baseline (no DR)
│   │   ├── train_udr.py        # Uniform Domain Randomization
│   │   ├── train_adr.py        # Automatic Domain Randomization
│   │   ├── train_ablation.py   # Part 2: Ablation study
│   │   └── train_optimal.py    # Part 2: Train with optimal params
│   ├── test/
│   │   └── test_comparison.py
│   ├── analysis/               # Part 2: Statistical analysis
│   │   ├── statistical_analysis.py
│   │   ├── plot_ablation.py
│   │   └── auto_select_params.py
│   └── run_ablation_study.sh   # Run all 8 configurations
├── logs/
├── configs/                    # Auto-generated optimal configs
├── docs/
│   ├── evaluation/
│   └── implementation/
│       ├── IMPLEMENTATION.md
│       └── PART2_IMPLEMENTATION.md
└── requirements.txt
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Train baseline (no randomization)
python scripts/train/train_baseline.py

# Train with Uniform Domain Randomization
python scripts/train/train_udr.py

# Train with ADR (2.5M, 5M, or 10M timesteps)
python scripts/train/train_adr.py --run 2_5M
python scripts/train/train_adr.py --run 5M
python scripts/train/train_adr.py --run 10M
```

### 3. Generate Comparison Charts

```bash
python scripts/test/test_comparison.py
```

### 4. Monitor Training

```bash
tensorboard --logdir ./logs/
```

## How ADR Works

1. **Start Simple**: Training begins with zero randomization (deterministic environment)
2. **Expand on Success**: When reward exceeds 1200, randomization range increases by 5%
3. **Contract on Failure**: When reward drops below 600, randomization range decreases
4. **Result**: Agent learns to handle maximum possible uncertainty

### Parameters Randomized

| Parameter | Range | Physical Meaning |
|-----------|-------|------------------|
| Mass | ±60% | Robot weight variation |
| Damping | ±60% | Joint resistance |
| Friction | ±60% | Ground grip |

## Reproducibility

All experiments use **seed=42** for reproducibility:
- NumPy random seed
- PyTorch random seed
- Environment reset seed
- PPO algorithm seed

## Documentation

- [Evaluation Report](docs/evaluation/EVALUATION_REPORT.md) - Full results analysis
- [Paper Draft](docs/evaluation/paper-draft.tex) - Research paper (LaTeX)
- [Implementation Guide](docs/implementation/IMPLEMENTATION.md) - Step-by-step guide

## References

- OpenAI et al., "Solving Rubik's Cube with a Robot Hand", 2019
- Mehta et al., "Active Domain Randomization", PMLR 2020
- Tan et al., "Sim-to-Real: Learning Agile Locomotion", RSS 2018

## License

See [LICENSE](LICENSE) for details.
