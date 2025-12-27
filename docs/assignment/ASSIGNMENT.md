# Robot Learning Project - Assignment

**Politecnico di Torino**  
A.Y. 2025-2026  
Course: Robot Learning | Lab: VANDAL

---

## Overview

### Topics Covered
- Sim-to-Real Transfer and Domain Randomization
- Hopper Environment
- Uniform Domain Randomization
- Project Extension

---

## 1. Sim-to-Real Transfer

### The Problem

**Goal:** Teach a robot how to push a box to a target location.

The Agent interacts with the environment via the standard RL loop:
1. **State** → 2. **Action** → 3. **Reward** → 4. **Next State**

| Environment | Role |
|-------------|------|
| **Simulation (Source)** | Training environment |
| **Real World (Target)** | Testing environment |

> **Reality Gap:** The mismatch between simulation and real-world dynamics that causes policies trained in simulation to fail in reality.

### The Solution: Domain Randomization

Training with **randomized dynamics** in simulation so the policy generalizes to real-world variations.

```
Simulation (Source)          →    Real World (Target)
[Randomized Parameters]            [Unknown Parameters]
    π_robust                           ✓ Works!
```

---

## 2. Uniform Domain Randomization (UDR)

By training across a **range of parameters** that encompasses the real-world target, the policy becomes robust:

```
           Table Friction
                ↑
                │  ┌─────────────────────┐
                │  │                     │
                │  │   UDR Training      │
                │  │   Region            │  ← Real-world
                │  │        ⊙            │    is inside
                │  │                     │
                │  └─────────────────────┘
                └────────────────────────→ Box Mass
```

---

## 3. Hopper Environment

**Goal:** Learn to hop forward with a one-legged robot without falling, while achieving the highest possible horizontal speed.

### Engine: MuJoCo
- Physics engine for detailed, efficient rigid body simulations with contacts
- Cross-platform GUI with interactive 3D visualization in OpenGL
- Advanced physics simulation

---

## 4. Core Part: Sim-to-Sim Transfer

### Step 1: Train the Agent

Train the Hopper agent with **PPO** or **SAC** algorithm.

| Environment | Description |
|-------------|-------------|
| **Source** | Torso mass is misspecified |
| **Target** | Correct dynamics for evaluation |

### Step 2: Implement UDR

Implement **Uniform Domain Randomization** for the link masses of the Hopper robot.

```
Source Environment                    Target Environment
├── Randomized link masses    →       ├── Fixed dynamics
├── Misspecified torso               └── Evaluate policy π
└── Train policy π
```

---

## 5. Project Extension

> **It's your turn!**
> 
> - Feel free to be **ambitious**
> - Let your work reflect a **research-inspired approach**

---

## 6. Installation

### Recommended: Use conda environments

### Required Libraries

```bash
pip install gymnasium
pip install mujoco
pip install "stable-baselines3[extra]>=2.0.0"
```

---

## 7. Project Exam

### Grading

| Component | Points |
|-----------|--------|
| Core part | 1.5 |
| Project extension | 4 |
| Oral exam | 20 |
| **Total** | **25.5** |

### Submission Details

#### Core Part (Individual)
- Submit to **personal repository**
- **Deadline:** January 8th, 23:59

#### Project Extension (Group)
- Submit to **team repository**
- Groups up to **4 people** allowed
- **Deadline:** One week before the exam call

### Oral Exam
- Team members must take the exam on the **same day**
- Project presentation with slides (~15 mins total, divided equally)
- Theory questions

---

**Politecnico di Torino** | A.Y. 2025-2026 | **VANDAL**