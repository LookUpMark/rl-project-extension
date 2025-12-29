#!/bin/bash
# Run all 10 ablation configurations

set -e
cd "$(dirname "$0")/.."

CONFIGS=(
    "baseline"      # No DR
    "udr"           # UDR ±30%
    "adr_none"      # ADR callback, no params
    "adr_mass"      # ADR mass only
    "adr_damp"      # ADR damping only
    "adr_fric"      # ADR friction only
    "adr_mass_damp" # ADR mass+damping
    "adr_mass_fric" # ADR mass+friction
    "adr_damp_fric" # ADR damping+friction
    "adr_all"       # ADR all params
)

echo "=== ABLATION STUDY (10 configurations) ==="
echo "Estimated time: ~25 hours"
echo ""

for config in "${CONFIGS[@]}"; do
    echo ">>> Running: $config"
    python train/train_ablation.py --config "$config"
    echo ">>> Completed: $config"
    echo ""
done

echo "=== ANALYSIS ==="
python analysis/statistical_analysis.py
python analysis/plot_ablation.py
python analysis/auto_select_params.py

echo ""
echo "✅ ABLATION COMPLETE"
echo "Results: logs/ablation/"
echo "Figures: docs/evaluation/figures/"
