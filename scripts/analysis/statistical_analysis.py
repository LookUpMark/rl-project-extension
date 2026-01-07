"""Statistical analysis for parameter relevance."""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

RESULTS_DIR = "logs/ablation/"


def load_results():
    """Load all ablation results."""
    results = []
    if not os.path.exists(RESULTS_DIR):
        return pd.DataFrame()
    for d in os.listdir(RESULTS_DIR):
        path = os.path.join(RESULTS_DIR, d, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return pd.DataFrame(results)


def compute_contributions(df):
    """Compute marginal contribution of each parameter."""
    contributions = {}
    for param in ['mass', 'damping', 'friction']:
        with_p = df[df['config'].apply(lambda x: x.get(param, False))]
        without_p = df[df['config'].apply(lambda x: not x.get(param, False))]
        
        if len(with_p) == 0 or len(without_p) == 0:
            contributions[param] = {'marginal_contribution': 0, 'p_value': 1.0, 'significant': False}
            continue
        
        contrib = with_p['transfer_gap'].mean() - without_p['transfer_gap'].mean()
        t, p = stats.ttest_ind(with_p['transfer_gap'], without_p['transfer_gap'])
        
        contributions[param] = {
            'marginal_contribution': contrib,
            'mean_with': with_p['transfer_gap'].mean(),
            'mean_without': without_p['transfer_gap'].mean(),
            't_statistic': t, 'p_value': p, 'significant': p < 0.05
        }
    return contributions


def compute_interactions(df):
    """Compute interaction effects between parameter pairs."""
    interactions = {}
    for p1, p2 in combinations(['mass', 'damping', 'friction'], 2):
        try:
            single1 = df[df['config'].apply(lambda x: x.get(p1) and not x.get(p2))]
            single2 = df[df['config'].apply(lambda x: not x.get(p1) and x.get(p2))]
            both = df[df['config'].apply(lambda x: x.get(p1) and x.get(p2))]
            base = df[df['config'].apply(lambda x: not x.get(p1) and not x.get(p2))]
            
            if len(base) == 0 or len(both) == 0: continue
            
            base_gap = base['transfer_gap'].mean()
            expected = single1['transfer_gap'].mean() + single2['transfer_gap'].mean() - base_gap
            actual = both['transfer_gap'].mean()
            
            interactions[f"{p1}_{p2}"] = {
                'expected_additive': expected, 'actual': actual,
                'interaction_effect': actual - expected, 'synergistic': actual > expected
            }
        except: continue
    return interactions


def rank_params(contributions):
    """Print parameter ranking."""
    ranking = sorted(contributions.items(), key=lambda x: abs(x[1]['marginal_contribution']), reverse=True)
    print("\n=== PARAMETER RANKING ===")
    for i, (p, d) in enumerate(ranking, 1):
        sig = "***" if d['p_value'] < 0.001 else "**" if d['p_value'] < 0.01 else "*" if d['p_value'] < 0.05 else ""
        print(f"{i}. {p.upper():10} | {d['marginal_contribution']:+.2f}% | p={d['p_value']:.4f} {sig}")
    return ranking


def save_report(df, contributions, interactions):
    """Save analysis report."""
    if len(df) == 0: return None
    
    report = {
        'summary': {
            'total': len(df),
            'best_config': df.loc[df['transfer_gap'].idxmax()]['config_name'],
            'best_gap': float(df['transfer_gap'].max()),
            'worst_config': df.loc[df['transfer_gap'].idxmin()]['config_name'],
            'worst_gap': float(df['transfer_gap'].min())
        },
        'parameter_contributions': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else str(vv) 
                                         for kk, vv in v.items()} for k, v in contributions.items()},
        'interaction_effects': interactions,
        'recommended_params': [p for p, d in contributions.items() if d['significant'] and d['marginal_contribution'] > 0]
    }
    
    with open('logs/ablation/analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: logs/ablation/analysis_report.json")
    return report


if __name__ == '__main__':
    print("=== STATISTICAL ANALYSIS ===\n")
    df = load_results()
    
    if len(df) == 0:
        print("No results found. Run ablation study first.")
        exit(1)
    
    print(f"Loaded {len(df)} configurations")
    contributions = compute_contributions(df)
    interactions = compute_interactions(df)
    rank_params(contributions)
    
    report = save_report(df, contributions, interactions)
    if report:
        print(f"\nRecommended: {report['recommended_params']}")
