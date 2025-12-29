"""Statistical analysis framework for parameter relevance."""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

RESULTS_DIR = "logs/ablation/"


def load_all_results():
    """Load results from all ablation configurations."""
    results = []
    if not os.path.exists(RESULTS_DIR):
        print(f"Warning: {RESULTS_DIR} not found")
        return pd.DataFrame()
    
    for config_dir in os.listdir(RESULTS_DIR):
        results_file = os.path.join(RESULTS_DIR, config_dir, "results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                results.append(json.load(f))
    return pd.DataFrame(results)


def compute_marginal_contributions(df):
    """Compute marginal contribution of each parameter."""
    params = ['mass', 'damping', 'friction']
    contributions = {}
    
    for param in params:
        with_param = df[df['config'].apply(lambda x: x.get(param, False))]
        without_param = df[df['config'].apply(lambda x: not x.get(param, False))]
        
        if len(with_param) == 0 or len(without_param) == 0:
            contributions[param] = {'marginal_contribution': 0, 'p_value': 1.0, 'significant': False}
            continue
        
        mean_with = with_param['transfer_gap'].mean()
        mean_without = without_param['transfer_gap'].mean()
        contribution = mean_with - mean_without
        
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
        try:
            single_p1 = df[(df['config'].apply(lambda x: x.get(p1))) & 
                           (~df['config'].apply(lambda x: x.get(p2)))]
            single_p2 = df[(~df['config'].apply(lambda x: x.get(p1))) & 
                           (df['config'].apply(lambda x: x.get(p2)))]
            combined = df[(df['config'].apply(lambda x: x.get(p1))) & 
                          (df['config'].apply(lambda x: x.get(p2)))]
            baseline = df[~df['config'].apply(lambda x: x.get(p1) or x.get(p2))]
            
            if len(baseline) == 0 or len(combined) == 0:
                continue
                
            base_gap = baseline['transfer_gap'].mean()
            expected = (single_p1['transfer_gap'].mean() - base_gap +
                       single_p2['transfer_gap'].mean() - base_gap + base_gap)
            actual = combined['transfer_gap'].mean()
            
            interactions[f"{p1}_{p2}"] = {
                'expected_additive': expected,
                'actual': actual,
                'interaction_effect': actual - expected,
                'synergistic': actual > expected
            }
        except:
            continue
    
    return interactions


def rank_parameters(contributions):
    """Rank parameters by their contribution."""
    ranking = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]['marginal_contribution']),
        reverse=True
    )
    
    print("\n=== PARAMETER RANKING ===")
    for i, (param, data) in enumerate(ranking, 1):
        sig = "***" if data['p_value'] < 0.001 else "**" if data['p_value'] < 0.01 else "*" if data['p_value'] < 0.05 else ""
        print(f"{i}. {param.upper():10} | Contribution: {data['marginal_contribution']:+.2f}% | p={data['p_value']:.4f} {sig}")
    
    return ranking


def generate_report(df, contributions, interactions):
    """Generate analysis report."""
    if len(df) == 0:
        print("No data to generate report")
        return None
        
    report = {
        'summary': {
            'total_configurations': len(df),
            'best_config': df.loc[df['transfer_gap'].idxmax()]['config_name'],
            'best_transfer_gap': float(df['transfer_gap'].max()),
            'worst_config': df.loc[df['transfer_gap'].idxmin()]['config_name'],
            'worst_transfer_gap': float(df['transfer_gap'].min())
        },
        'parameter_contributions': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                                         for kk, vv in v.items()} 
                                    for k, v in contributions.items()},
        'interaction_effects': interactions,
        'recommended_params': [p for p, d in contributions.items() 
                              if d['significant'] and d['marginal_contribution'] > 0]
    }
    
    os.makedirs('logs/ablation', exist_ok=True)
    with open('logs/ablation/analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nReport saved to logs/ablation/analysis_report.json")
    return report


if __name__ == '__main__':
    print("=== STATISTICAL ANALYSIS: Parameter Relevance ===\n")
    
    df = load_all_results()
    if len(df) == 0:
        print("No ablation results found. Run ablation experiments first.")
        exit(1)
        
    print(f"Loaded {len(df)} configurations\n")
    
    contributions = compute_marginal_contributions(df)
    interactions = compute_interaction_effects(df)
    ranking = rank_parameters(contributions)
    
    report = generate_report(df, contributions, interactions)
    
    if report:
        print("\n=== RECOMMENDED PARAMETERS ===")
        print(f"Based on statistical significance: {report['recommended_params']}")
