# Part 2: Parameter Relevance Analysis & Automated Selection

**Version:** 1.0  
**Status:** ✅ COMPLETED (December 29, 2024)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Configurations | 10 |
| Best Configuration | `adr_fric` (+154.6% transfer gap) |
| Most Relevant Parameter | **FRICTION** (+68.7% contribution) |
| Total Training Time | ~25 hours |
| Seed | 42 (reproducible) |

---

## 1. Overview

This part extends the ADR implementation with:

1. **Ablation Study** - Systematic analysis of how performance varies with different parameter combinations
2. **Statistical Analysis** - Framework to quantify parameter relevance
3. **Automated Selection** - System to automatically choose parameters based on data

---

## 2. Ablation Study Design

### 2.1 Configurations (10 Total)

| Config | Mass | Damping | Friction | Mode |
|--------|------|---------|----------|------|
| baseline | ✗ | ✗ | ✗ | baseline |
| udr | ✓ | ✓ | ✓ | udr |
| adr_none | ✗ | ✗ | ✗ | adr |
| adr_mass | ✓ | ✗ | ✗ | adr |
| adr_damp | ✗ | ✓ | ✗ | adr |
| adr_fric | ✗ | ✗ | ✓ | adr |
| adr_mass_damp | ✓ | ✓ | ✗ | adr |
| adr_mass_fric | ✓ | ✗ | ✓ | adr |
| adr_damp_fric | ✗ | ✓ | ✓ | adr |
| adr_all | ✓ | ✓ | ✓ | adr |

### 2.2 Running the Ablation Study

```bash
# Run all configurations
for config in baseline udr adr_none adr_mass adr_damp adr_fric \
              adr_mass_damp adr_mass_fric adr_damp_fric adr_all; do
    python scripts/train/train_ablation.py --config $config
done
```

Each run:
- 2.5M timesteps
- Seed = 42
- Saves to `logs/ablation/{config}/results.json`

---

## 3. Statistical Analysis

### 3.1 Marginal Contributions

For each parameter P, we compute:

```
Contribution(P) = mean(gap | P=on) - mean(gap | P=off)
```

### 3.2 Running Analysis

```bash
python scripts/analysis/statistical_analysis.py
```

Output: `logs/ablation/analysis_report.json`

### 3.3 Results

| Rank | Parameter | Contribution | p-value | Significant? |
|------|-----------|--------------|---------|--------------|
| 1 | **FRICTION** | **+68.70%** | 0.0744 | * (p<0.10) |
| 2 | MASS | -15.68% | 0.7120 | No |
| 3 | DAMPING | -0.86% | 0.9840 | No |

---

## 4. Interaction Effects

Discovered strong antagonistic interactions:

| Interaction | Expected | Actual | Effect |
|-------------|----------|--------|--------|
| M × D | -11.1% | -6.1% | +5.0% (synergistic) |
| M × F | +97.5% | +2.9% | **-94.5%** (antagonistic) |
| D × F | +113.0% | +6.8% | **-106.2%** (antagonistic) |

**Key Finding:** Adding mass to friction reduces friction's positive effect by 94.5%!

---

## 5. Automated Parameter Selection

### 5.1 Usage

```bash
python scripts/analysis/auto_select_params.py
```

### 5.2 Selection Strategies

| Strategy | Logic |
|----------|-------|
| `significant` | Only statistically significant positive contributors |
| `positive` | All positive contributors |
| `top_N` | Top N by magnitude |

### 5.3 Output

Generates `configs/optimal_adr.json`:
```json
{
  "adr_enabled_params": {"mass": false, "damping": false, "friction": true},
  "selection_method": "statistical_significance"
}
```

---

## 6. Visualization

```bash
python scripts/analysis/plot_ablation.py
```

Generates:
- `ablation_transfer_gap.png` - Bar chart of all configs
- `mode_comparison.png` - Baseline vs UDR vs Best ADR
- `parameter_analysis.png` - Matrix + scatter plot
- `individual_param_impact.png` - Box plots per parameter

---

## 7. Key Findings

1. **FRICTION is king** - Friction-only ADR achieves +154.6% transfer gap
2. **MASS can hurt** - Mass randomization contributes -15.7% 
3. **Interactions matter** - Combining parameters can reduce effectiveness
4. **Selective > Uniform** - Data-driven selection outperforms uniform randomization

---

## 8. Practical Recommendations

| Goal | Recommended Config |
|------|-------------------|
| Maximum transfer | `adr_fric` (friction only) |
| Balanced robustness | `adr_damp_fric` |
| Conservative approach | `udr` (fixed ±30%) |
| **Avoid** | `adr_mass`, `adr_mass_damp` |
