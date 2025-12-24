# Sim-to-Real Hopper with Automatic Domain Randomization (ADR)

An advanced reinforcement learning project implementing **Automatic Domain Randomization** for robust Sim-to-Real transfer in the MuJoCo Hopper environment.

## Overview

This project extends the standard Hopper-v4 environment to implement ADR (Automatic Domain Randomization), a curriculum-based technique that automatically adjusts the difficulty of domain randomization based on agent performance. This produces policies that are robust to the reality gap without manual hyperparameter tuning.

### Key Features

- **Adaptive Difficulty**: Environment complexity increases when the agent performs well and decreases when it struggles
- **Multi-Parameter Randomization**: Randomizes mass, joint damping, and ground friction
- **Tensorboard Logging**: Track ADR evolution during training
- **Sim-to-Real Ready**: Designed for transfer to real robotic systems

## Project Structure

```
├── env/
│   ├── __init__.py
│   ├── custom_hopper.py     # Extended Hopper with ADR support
│   └── assets/
│       └── hopper.xml       # MuJoCo model
├── callbacks/
│   ├── __init__.py
│   └── adr_callback.py      # ADR training callback
├── notebooks/
│   └── verification/
│       ├── verify-member-1.ipynb  # Member 1 implementation verification
│       └── verify-member-2.ipynb  # Member 2 implementation verification
├── logs/                    # Tensorboard logs directory
├── train.py                 # Main training script
├── test_random_policy.py    # Environment testing script
├── requirements.txt
└── docs/
    └── implementation/
        ├── IMPLEMENTATION.md  # Full implementation guide (team overview)
        ├── MEMBER-2.md        # Detailed guide for Member 2
        └── REPORT.md          # Research report
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Test Environment

```bash
python test_random_policy.py
```

### 3. Train with ADR

```bash
python train.py
```

### 4. Monitor Training

```bash
tensorboard --logdir ./logs/
```

## How ADR Works

1. **Start Simple**: Training begins with zero randomization (deterministic environment)
2. **Expand on Success**: When reward exceeds threshold, randomization range increases
3. **Contract on Failure**: When reward drops, randomization range decreases
4. **Result**: Agent learns to handle maximum possible uncertainty

## Documentation

- [Implementation Guide](docs/implementation/IMPLEMENTATION.md) - Complete step-by-step guide
- [Member 2 Guide](docs/implementation/MEMBER-2.md) - Detailed guide for Integration & DevOps
- [Research Report](docs/implementation/REPORT.md) - Theoretical background and analysis
- [Member 1 Verification](notebooks/verification/verify-member-1.ipynb) - Verify Member 1 implementation
- [Member 2 Verification](notebooks/verification/verify-member-2.ipynb) - Verify Member 2 implementation

## References

- OpenAI et al., "Solving Rubik's Cube with a Robot Hand", 2019
- Mehta et al., "Active Domain Randomization", PMLR 2020
- Tan et al., "Sim-to-Real: Learning Agile Locomotion", RSS 2018

## License

See [LICENSE](LICENSE) for details.
