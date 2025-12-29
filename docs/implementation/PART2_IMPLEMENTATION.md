# PART 2: Parameter Relevance Analysis & Automated Hyperparameter Selection

**Versione:** 1.0  
**Obiettivo:** Analisi statistica della rilevanza dei parametri ADR e automazione della selezione iperparametri  
**Team:** 2 Membri con ruoli bilanciati  
**Status:** ✅ **COMPLETED** (December 29, 2024)

---

## EXECUTION SUMMARY

> **All experiments completed successfully.** See [PART2_EVALUATION_REPORT.md](../evaluation/PART2_EVALUATION_REPORT.md) for detailed results.

| Metric | Value |
|--------|-------|
| Total Configurations Trained | 10 |
| Best Configuration | `adr_fric` (+154.6% transfer gap) |
| Most Relevant Parameter | **FRICTION** (+68.7% contribution) |
| Total Training Time | ~5 hours |
| Seed | 42 (reproducible) |

---

## PANORAMICA

Questa seconda parte del progetto estende l'implementazione ADR con:

1. **Ablation Study**: Analisi sistematica di come le performance variano randomizzando N parametri vs M, e del contributo specifico di ogni parametro
2. **Statistical Analysis**: Framework per quantificare la rilevanza statistica di ogni parametro
3. **Automated Hyperparameter Selection**: Sistema che automatizza la scelta dei parametri da randomizzare in base alla loro rilevanza

---

## SUDDIVISIONE MEMBRI

| Membro | Ruolo | Responsabilità |
|--------|-------|----------------|
| **Membro 1** | Core Analysis & Experiments | Ablation study, training configurations, data collection |
| **Membro 2** | Statistical Tools & Automation | Framework statistico, visualizzazioni, sistema di auto-selection |

---

# MEMBRO 1: Core Analysis & Experiments

## Fase 1.1: Ablation Study - Configurazioni di Training

### Obiettivo
Creare un set di esperimenti che testi sistematicamente diverse combinazioni di parametri randomizzati.

### Configurazioni da Implementare

| Config ID | Massa | Damping | Friction | Descrizione |
|-----------|-------|---------|----------|-------------|
| `none` | ❌ | ❌ | ❌ | Baseline (no randomization) |
| `mass_only` | ✅ | ❌ | ❌ | Solo massa |
| `damp_only` | ❌ | ✅ | ❌ | Solo damping |
| `fric_only` | ❌ | ❌ | ✅ | Solo friction |
| `mass_damp` | ✅ | ✅ | ❌ | Massa + Damping |
| `mass_fric` | ✅ | ❌ | ✅ | Massa + Friction |
| `damp_fric` | ❌ | ✅ | ✅ | Damping + Friction |
| `all` | ✅ | ✅ | ✅ | Tutti (baseline ADR) |

### File da Creare: `scripts/train/train_ablation.py`

```python
"""Ablation study training script - tests individual parameter contributions."""
import argparse
import gymnasium as gym
import numpy as np
import torch
import random
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *
from callbacks.adr_callback import ADRCallback

SEED = 42
TIMESTEPS = 2_500_000  # Standard run length

# Ablation configurations: which parameters to randomize
ABLATION_CONFIGS = {
    'none':      {'mass': False, 'damping': False, 'friction': False},
    'mass_only': {'mass': True,  'damping': False, 'friction': False},
    'damp_only': {'mass': False, 'damping': True,  'friction': False},
    'fric_only': {'mass': False, 'damping': False, 'friction': True},
    'mass_damp': {'mass': True,  'damping': True,  'friction': False},
    'mass_fric': {'mass': True,  'damping': False, 'friction': True},
    'damp_fric': {'mass': False, 'damping': True,  'friction': True},
    'all':       {'mass': True,  'damping': True,  'friction': True},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config_name):
    set_seed(SEED)
    
    config = ABLATION_CONFIGS[config_name]
    log_dir = f"./logs/ablation/{config_name}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment with specific ADR config
    env_source = Monitor(gym.make('CustomHopper-source-v0', udr=False))
    env_source.reset(seed=SEED)
    
    # Set which parameters are active for ADR
    env_unwrapped = env_source.unwrapped
    env_unwrapped.adr_enabled_params = config  # New attribute to implement
    
    env_target = gym.make('CustomHopper-target-v0', udr=False)
    env_target.reset(seed=SEED)
    
    print(f'=== ABLATION: {config_name} (Seed: {SEED}) ===')
    print(f'Config: {config}')
    
    # Train with ADR callback
    adr_callback = ADRCallback(check_freq=2048)
    model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log=log_dir, seed=SEED)
    model.learn(total_timesteps=TIMESTEPS, callback=adr_callback, progress_bar=True)
    
    # Save model
    model.save(f"{log_dir}ppo_ablation_{config_name}")
    
    # Evaluate
    mean_src, std_src = evaluate_policy(model, env_source, n_eval_episodes=50)
    mean_tgt, std_tgt = evaluate_policy(model, env_target, n_eval_episodes=50)
    
    gap = (mean_tgt - mean_src) / mean_src * 100 if mean_src > 0 else 0
    final_adr = env_unwrapped.get_adr_info()
    
    # Save results to JSON
    results = {
        'config_name': config_name,
        'config': config,
        'seed': SEED,
        'timesteps': TIMESTEPS,
        'source_mean': float(mean_src),
        'source_std': float(std_src),
        'target_mean': float(mean_tgt),
        'target_std': float(std_tgt),
        'transfer_gap': float(gap),
        'final_adr_state': final_adr,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{log_dir}results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSource: {mean_src:.2f} ± {std_src:.2f}")
    print(f"Target: {mean_tgt:.2f} ± {std_tgt:.2f}")
    print(f"Transfer gap: {gap:+.1f}%")
    
    env_source.close()
    env_target.close()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--config', type=str, required=True, 
                        choices=list(ABLATION_CONFIGS.keys()))
    args = parser.parse_args()
    main(args.config)
```

---

## Fase 1.2: Modifica Environment per Ablation

### File da Modificare: `env/custom_hopper.py`

Aggiungere supporto per abilitare/disabilitare parametri specifici:

```python
# In __init__, dopo adr_state:
self.adr_enabled_params = {'mass': True, 'damping': True, 'friction': True}

# Modificare sample_parameters():
def sample_parameters(self):
    """Sample parameters based on ADR state and enabled params."""
    mass_range = self.adr_state["mass_range"] if self.adr_enabled_params.get('mass', True) else 0.0
    damping_range = self.adr_state["damping_range"] if self.adr_enabled_params.get('damping', True) else 0.0
    friction_range = self.adr_state["friction_range"] if self.adr_enabled_params.get('friction', True) else 0.0
    
    # ... rest of sampling logic with conditional ranges ...

# Modificare update_adr():
def update_adr(self, mean_reward, low_th, high_th):
    """Update only enabled ADR parameters."""
    status = "stable"
    if mean_reward >= high_th:
        for k in self.adr_state:
            param_key = k.replace('_range', '')
            if self.adr_enabled_params.get(param_key, True):
                self.adr_state[k] = min(self.adr_state[k] + self.adr_step_size, 1.0)
        status = "expanded"
    elif mean_reward < low_th:
        for k in self.adr_state:
            param_key = k.replace('_range', '')
            if self.adr_enabled_params.get(param_key, True):
                self.adr_state[k] = max(self.adr_state[k] - self.adr_step_size, 0.0)
        status = "contracted"
    return status, self.adr_state.copy()
```

---

## Fase 1.3: Script di Esecuzione Completa

### File da Creare: `scripts/run_ablation_study.sh`

```bash
#!/bin/bash
# Run all ablation configurations

CONFIGS=("none" "mass_only" "damp_only" "fric_only" "mass_damp" "mass_fric" "damp_fric" "all")

echo "=== ABLATION STUDY: Starting all configurations ==="
echo "Estimated time: ~20 hours (8 configs × 2.5 hours)"

for config in "${CONFIGS[@]}"; do
    echo ""
    echo ">>> Running: $config"
    python scripts/train/train_ablation.py --config $config
    echo ">>> Completed: $config"
done

echo ""
echo "=== ABLATION STUDY COMPLETE ==="
echo "Results saved in logs/ablation/*/results.json"
```

---

## Fase 1.4: Raccolta Dati

### Dati da Registrare per Ogni Configurazione

| Metrica | Descrizione |
|---------|-------------|
| `source_mean` | Reward medio su source |
| `target_mean` | Reward medio su target |
| `transfer_gap` | Differenza percentuale |
| `final_adr_mass` | Range finale massa |
| `final_adr_damping` | Range finale damping |
| `final_adr_friction` | Range finale friction |
| `training_time` | Tempo di training |
| `convergence_step` | Step a cui ADR si stabilizza |

---

# MEMBRO 2: Statistical Tools & Automation

## Fase 2.1: Framework di Analisi Statistica

### File da Creare: `scripts/analysis/statistical_analysis.py`

```python
"""Statistical analysis framework for parameter relevance."""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "logs/ablation/"


def load_all_results():
    """Load results from all ablation configurations."""
    results = []
    for config_dir in os.listdir(RESULTS_DIR):
        results_file = os.path.join(RESULTS_DIR, config_dir, "results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                results.append(json.load(f))
    return pd.DataFrame(results)


def compute_marginal_contributions(df):
    """Compute marginal contribution of each parameter.
    
    For each parameter P, computes:
    - Mean improvement when P is added to configurations without P
    - Statistical significance via paired t-test
    """
    params = ['mass', 'damping', 'friction']
    contributions = {}
    
    for param in params:
        # Configurations with and without this param
        with_param = df[df['config'].apply(lambda x: x.get(param, False))]
        without_param = df[df['config'].apply(lambda x: not x.get(param, False))]
        
        # Marginal contribution to transfer gap
        mean_with = with_param['transfer_gap'].mean()
        mean_without = without_param['transfer_gap'].mean()
        contribution = mean_with - mean_without
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(
            with_param['transfer_gap'].values,
            without_param['transfer_gap'].values
        )
        
        contributions[param] = {
            'marginal_contribution': contribution,
            'mean_with_param': mean_with,
            'mean_without_param': mean_without,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    return contributions


def compute_interaction_effects(df):
    """Compute interaction effects between pairs of parameters."""
    params = ['mass', 'damping', 'friction']
    interactions = {}
    
    for p1, p2 in combinations(params, 2):
        # Expected additive effect vs actual combined effect
        single_p1 = df[(df['config'].apply(lambda x: x.get(p1))) & 
                       (df['config'].apply(lambda x: not x.get(p2)))]
        single_p2 = df[(df['config'].apply(lambda x: not x.get(p1))) & 
                       (df['config'].apply(lambda x: x.get(p2)))]
        combined = df[(df['config'].apply(lambda x: x.get(p1))) & 
                      (df['config'].apply(lambda x: x.get(p2)))]
        baseline = df[~df['config'].apply(lambda x: x.get(p1) or x.get(p2))]
        
        expected = (single_p1['transfer_gap'].mean() - baseline['transfer_gap'].mean() +
                   single_p2['transfer_gap'].mean() - baseline['transfer_gap'].mean() +
                   baseline['transfer_gap'].mean())
        actual = combined['transfer_gap'].mean()
        
        interactions[f"{p1}_{p2}"] = {
            'expected_additive': expected,
            'actual': actual,
            'interaction_effect': actual - expected,
            'synergistic': actual > expected
        }
    
    return interactions


def rank_parameters(contributions):
    """Rank parameters by their contribution to transfer performance."""
    ranking = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]['marginal_contribution']),
        reverse=True
    )
    
    print("\n=== PARAMETER RANKING (by marginal contribution) ===")
    for i, (param, data) in enumerate(ranking, 1):
        sig = "***" if data['p_value'] < 0.001 else "**" if data['p_value'] < 0.01 else "*" if data['p_value'] < 0.05 else ""
        print(f"{i}. {param.upper():12} | Contribution: {data['marginal_contribution']:+.2f}% | p={data['p_value']:.4f} {sig}")
    
    return ranking


def generate_report(df, contributions, interactions):
    """Generate comprehensive analysis report."""
    report = {
        'summary': {
            'total_configurations': len(df),
            'best_config': df.loc[df['transfer_gap'].idxmax()]['config_name'],
            'best_transfer_gap': df['transfer_gap'].max(),
            'worst_config': df.loc[df['transfer_gap'].idxmin()]['config_name'],
            'worst_transfer_gap': df['transfer_gap'].min()
        },
        'parameter_contributions': contributions,
        'interaction_effects': interactions,
        'recommended_params': [p for p, d in contributions.items() if d['significant'] and d['marginal_contribution'] > 0]
    }
    
    with open('logs/ablation/analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    return report


if __name__ == '__main__':
    print("=== STATISTICAL ANALYSIS: Parameter Relevance ===\n")
    
    df = load_all_results()
    print(f"Loaded {len(df)} configurations\n")
    
    contributions = compute_marginal_contributions(df)
    interactions = compute_interaction_effects(df)
    ranking = rank_parameters(contributions)
    
    report = generate_report(df, contributions, interactions)
    
    print("\n=== RECOMMENDED PARAMETERS ===")
    print(f"Based on statistical significance: {report['recommended_params']}")
```

---

## Fase 2.2: Visualizzazioni

### File da Creare: `scripts/analysis/plot_ablation.py`

```python
"""Generate ablation study visualizations."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "docs/evaluation/figures/"


def load_results():
    results = []
    for config_dir in os.listdir("logs/ablation/"):
        path = f"logs/ablation/{config_dir}/results.json"
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return pd.DataFrame(results)


def plot_transfer_gap_comparison(df):
    """Bar chart comparing transfer gaps across configurations."""
    plt.figure(figsize=(12, 6))
    
    # Sort by transfer gap
    df_sorted = df.sort_values('transfer_gap', ascending=False)
    
    colors = ['#2ca02c' if g > 0 else '#d62728' for g in df_sorted['transfer_gap']]
    
    plt.barh(df_sorted['config_name'], df_sorted['transfer_gap'], color=colors)
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Transfer Gap (%)')
    plt.ylabel('Configuration')
    plt.title('Ablation Study: Transfer Gap by Parameter Configuration')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}ablation_transfer_gap.png', dpi=150)
    plt.close()


def plot_parameter_contribution_heatmap(df):
    """Heatmap showing parameter contributions."""
    # Create binary matrix for each config
    params = ['mass', 'damping', 'friction']
    matrix = np.zeros((len(df), len(params)))
    
    for i, row in df.iterrows():
        for j, p in enumerate(params):
            matrix[i, j] = row['config'].get(p, False)
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(matrix, columns=params, index=df['config_name'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Config matrix
    sns.heatmap(heatmap_df, annot=True, cmap='RdYlGn', ax=ax1, cbar=False)
    ax1.set_title('Parameter Configuration Matrix')
    
    # Transfer gap by number of params
    n_params = heatmap_df.sum(axis=1)
    df['n_params'] = n_params.values
    
    ax2.scatter(df['n_params'], df['transfer_gap'], s=100)
    for i, row in df.iterrows():
        ax2.annotate(row['config_name'], (row['n_params'], row['transfer_gap']), 
                    fontsize=8, ha='center', va='bottom')
    ax2.set_xlabel('Number of Randomized Parameters')
    ax2.set_ylabel('Transfer Gap (%)')
    ax2.set_title('Transfer Gap vs Number of Parameters')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}parameter_analysis.png', dpi=150)
    plt.close()


def plot_individual_param_impact(df):
    """Box plots showing impact of each parameter."""
    params = ['mass', 'damping', 'friction']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, param in zip(axes, params):
        with_param = df[df['config'].apply(lambda x: x.get(param, False))]['transfer_gap']
        without_param = df[df['config'].apply(lambda x: not x.get(param, False))]['transfer_gap']
        
        data = [without_param, with_param]
        bp = ax.boxplot(data, labels=[f'No {param}', f'With {param}'])
        ax.set_ylabel('Transfer Gap (%)')
        ax.set_title(f'{param.capitalize()} Impact')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('Individual Parameter Impact on Transfer Performance', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}individual_param_impact.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_results()
    
    print("Generating ablation visualizations...")
    plot_transfer_gap_comparison(df)
    plot_parameter_contribution_heatmap(df)
    plot_individual_param_impact(df)
    print(f"Saved to {OUTPUT_DIR}")
```

---

## Fase 2.3: Sistema di Auto-Selection

### File da Creare: `scripts/analysis/auto_select_params.py`

```python
"""Automated parameter selection based on statistical analysis."""
import json
import numpy as np
from statistical_analysis import load_all_results, compute_marginal_contributions


class ParameterSelector:
    """Selects optimal parameters based on ablation study results."""
    
    def __init__(self, results_dir="logs/ablation/"):
        self.results_dir = results_dir
        self.contributions = None
        self.selection = None
    
    def analyze(self):
        """Run statistical analysis on ablation results."""
        df = load_all_results()
        self.contributions = compute_marginal_contributions(df)
        return self
    
    def select(self, strategy='significant', threshold=0.05):
        """Select parameters based on strategy.
        
        Strategies:
        - 'significant': Select statistically significant positive contributors
        - 'positive': Select all positive contributors
        - 'top_n': Select top N contributors by magnitude
        """
        if self.contributions is None:
            self.analyze()
        
        if strategy == 'significant':
            self.selection = {
                p: True for p, d in self.contributions.items()
                if d['p_value'] < threshold and d['marginal_contribution'] > 0
            }
        elif strategy == 'positive':
            self.selection = {
                p: True for p, d in self.contributions.items()
                if d['marginal_contribution'] > 0
            }
        elif strategy.startswith('top_'):
            n = int(strategy.split('_')[1])
            sorted_params = sorted(
                self.contributions.items(),
                key=lambda x: x[1]['marginal_contribution'],
                reverse=True
            )
            self.selection = {p: True for p, _ in sorted_params[:n]}
        
        # Fill in non-selected params
        for p in ['mass', 'damping', 'friction']:
            if p not in self.selection:
                self.selection[p] = False
        
        return self.selection
    
    def get_recommendation(self):
        """Generate human-readable recommendation."""
        if self.selection is None:
            self.select()
        
        enabled = [p for p, v in self.selection.items() if v]
        
        return {
            'enabled_params': enabled,
            'config': self.selection,
            'rationale': {p: self.contributions[p] for p in enabled}
        }
    
    def export_config(self, output_path="configs/optimal_adr.json"):
        """Export selection as ADR configuration file."""
        recommendation = self.get_recommendation()
        
        config = {
            'adr_enabled_params': recommendation['config'],
            'selection_method': 'statistical_significance',
            'based_on_ablation': True,
            'contributions': {
                p: {
                    'contribution': d['marginal_contribution'],
                    'p_value': d['p_value']
                }
                for p, d in self.contributions.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Optimal configuration saved to {output_path}")
        return config


if __name__ == '__main__':
    selector = ParameterSelector()
    recommendation = selector.analyze().get_recommendation()
    
    print("=== AUTOMATED PARAMETER SELECTION ===")
    print(f"Recommended params: {recommendation['enabled_params']}")
    print(f"Config: {recommendation['config']}")
    
    selector.export_config()
```

---

## Fase 2.4: Training con Parametri Ottimali

### File da Creare: `scripts/train/train_optimal.py`

```python
"""Train with automatically selected optimal parameters."""
import json
import gymnasium as gym
import numpy as np
import torch
import random
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *
from callbacks.adr_callback import ADRCallback

SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config_path="configs/optimal_adr.json", timesteps=5_000_000):
    set_seed(SEED)
    
    # Load optimal config
    with open(config_path) as f:
        config = json.load(f)
    
    enabled_params = config['adr_enabled_params']
    
    log_dir = "./logs/optimal/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment with optimal config
    env_source = Monitor(gym.make('CustomHopper-source-v0', udr=False))
    env_source.reset(seed=SEED)
    env_source.unwrapped.adr_enabled_params = enabled_params
    
    env_target = gym.make('CustomHopper-target-v0', udr=False)
    env_target.reset(seed=SEED)
    
    print(f'=== OPTIMAL ADR TRAINING (Seed: {SEED}) ===')
    print(f'Enabled params: {enabled_params}')
    print(f'Timesteps: {timesteps:,}')
    
    # Train
    adr_callback = ADRCallback(check_freq=2048)
    model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log=log_dir, seed=SEED)
    model.learn(total_timesteps=timesteps, callback=adr_callback, progress_bar=True)
    
    # Save
    model.save(f"{log_dir}ppo_hopper_optimal")
    
    # Evaluate
    mean_src, std_src = evaluate_policy(model, env_source, n_eval_episodes=50)
    mean_tgt, std_tgt = evaluate_policy(model, env_target, n_eval_episodes=50)
    
    gap = (mean_tgt - mean_src) / mean_src * 100 if mean_src > 0 else 0
    
    print(f"\nSource: {mean_src:.2f} ± {std_src:.2f}")
    print(f"Target: {mean_tgt:.2f} ± {std_tgt:.2f}")
    print(f"Transfer gap: {gap:+.1f}%")
    
    env_source.close()
    env_target.close()


if __name__ == '__main__':
    main()
```

---

# TIMELINE SUGGERITA

| Giorno | Membro 1 | Membro 2 |
|--------|----------|----------|
| **1** | Implementare modifiche `custom_hopper.py` | Setup struttura cartelle, init scripts |
| **2** | Creare `train_ablation.py` | Creare `statistical_analysis.py` |
| **3** | Eseguire configs: none, mass_only, damp_only, fric_only | Creare `plot_ablation.py` |
| **4** | Eseguire configs: mass_damp, mass_fric, damp_fric, all | Creare `auto_select_params.py` |
| **5** | Verificare risultati, raccogliere JSON | Eseguire analisi statistica |
| **6** | Training con config ottimale | Generare visualizzazioni e report |
| **7** | Valutazione finale, confronto con Part 1 | Integrare tutto, documentazione finale |

---

# VERIFICA

## Test Membro 1

```bash
# Test singola configurazione ablation
python scripts/train/train_ablation.py --config mass_only

# Verificare che crei:
# - logs/ablation/mass_only/PPO_1/
# - logs/ablation/mass_only/ppo_ablation_mass_only.zip
# - logs/ablation/mass_only/results.json
```

## Test Membro 2

```bash
# Test analisi (richiede almeno 2 configurazioni completate)
python scripts/analysis/statistical_analysis.py

# Verificare output:
# - logs/ablation/analysis_report.json
# - Ranking stampato a console

# Test visualizzazioni
python scripts/analysis/plot_ablation.py

# Verificare:
# - docs/evaluation/figures/ablation_transfer_gap.png
# - docs/evaluation/figures/parameter_analysis.png
```

---

# OUTPUT ATTESI

## Domande di Ricerca Risposte

1. **Quale parametro contribuisce di più al transfer?**
   - Ranking quantitativo con p-values
   
2. **Esistono interazioni tra parametri?**
   - Analisi effetti sinergici vs additivi
   
3. **Qual è la configurazione ottimale?**
   - Raccomandazione automatica basata sui dati

## Deliverables

| Deliverable | Responsabile |
|-------------|--------------|
| 8 modelli ablation trainati | Membro 1 |
| JSON con risultati strutturati | Membro 1 |
| Report analisi statistica | Membro 2 |
| Grafici comparativi | Membro 2 |
| Sistema auto-selection | Membro 2 |
| Modello con config ottimale | Entrambi |
| Sezione paper Part 2 | Entrambi |
