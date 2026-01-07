"""Generate ablation study visualizations."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

OUTPUT_DIR = "docs/evaluation/figures/"
RESULTS_DIR = "logs/ablation/"


def load_results():
    """Load all ablation results."""
    results = []
    if not os.path.exists(RESULTS_DIR): return pd.DataFrame()
    for d in os.listdir(RESULTS_DIR):
        path = os.path.join(RESULTS_DIR, d, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return pd.DataFrame(results)


def get_color(mode):
    return '#1f77b4' if mode == 'baseline' else '#ff7f0e' if mode == 'udr' else '#2ca02c'


def plot_transfer_gaps(df):
    """Horizontal bar chart of transfer gaps."""
    plt.figure(figsize=(14, 7))
    df_sorted = df.sort_values('transfer_gap', ascending=True)
    colors = [get_color(m) for m in df_sorted['mode']]
    
    bars = plt.barh(df_sorted['config_name'], df_sorted['transfer_gap'], color=colors)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlabel('Transfer Gap (%)'); plt.ylabel('Configuration')
    plt.title('Ablation Study: Transfer Gap')
    
    for bar, val in zip(bars, df_sorted['transfer_gap']):
        plt.text(val + 1 if val >= 0 else val - 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    
    plt.legend(handles=[Patch(color='#1f77b4', label='Baseline'),
                        Patch(color='#ff7f0e', label='UDR'),
                        Patch(color='#2ca02c', label='ADR')], loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}ablation_transfer_gap.png', dpi=150)
    plt.close()
    print("Saved: ablation_transfer_gap.png")


def plot_mode_comparison(df):
    """Compare Baseline vs UDR vs Best ADR."""
    adr_df = df[df['mode'] == 'adr']
    best_adr = adr_df.loc[adr_df['transfer_gap'].idxmax()] if len(adr_df) > 0 else None
    
    data = []
    for mode in ['baseline', 'udr']:
        m_df = df[df['mode'] == mode]
        if len(m_df) > 0:
            row = m_df.iloc[0]
            data.append({'mode': mode.upper(), 'source': row['source_mean'],
                        'target': row['target_mean'], 'gap': row['transfer_gap']})
    
    if best_adr is not None:
        data.append({'mode': f"ADR ({best_adr['config_name']})", 'source': best_adr['source_mean'],
                    'target': best_adr['target_mean'], 'gap': best_adr['transfer_gap']})
    
    if not data: return
    result = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(result)); w = 0.35
    
    ax1.bar(x-w/2, result['source'], w, label='Source', color='#2ca02c')
    ax1.bar(x+w/2, result['target'], w, label='Target', color='#d62728')
    ax1.set_xticks(x); ax1.set_xticklabels(result['mode'], rotation=15, ha='right')
    ax1.set(ylabel='Reward', title='Performance'); ax1.legend(); ax1.grid(axis='y', alpha=0.3)
    
    colors = ['#1f77b4' if 'BASELINE' in m else '#ff7f0e' if 'UDR' in m else '#2ca02c' for m in result['mode']]
    ax2.bar(result['mode'], result['gap'], color=colors)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.set(ylabel='Gap (%)', title='Transfer'); ax2.set_xticklabels(result['mode'], rotation=15, ha='right')
    for i, g in enumerate(result['gap']):
        ax2.text(i, g + 1, f'{g:+.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}mode_comparison.png', dpi=150)
    plt.close()
    print("Saved: mode_comparison.png")


def plot_param_analysis(df):
    """Parameter matrix and scatter plot."""
    df = df.copy()
    for p in ['mass', 'damping', 'friction']:
        df[f'{p}_on'] = df['config'].apply(lambda x: x.get(p, False))
    df['n_params'] = df[['mass_on', 'damping_on', 'friction_on']].sum(axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    heatmap = df[['mass_on', 'damping_on', 'friction_on']].astype(int)
    heatmap.columns = ['mass', 'damping', 'friction']
    heatmap.index = df['config_name']
    sns.heatmap(heatmap, annot=True, cmap='RdYlGn', ax=ax1, cbar=False)
    ax1.set_title('Parameter Matrix')
    
    colors = df['mode'].apply(get_color)
    ax2.scatter(df['n_params'], df['transfer_gap'], s=100, c=colors, edgecolors='black')
    for _, row in df.iterrows():
        ax2.annotate(row['config_name'], (row['n_params'], row['transfer_gap']), fontsize=8, ha='center', va='bottom')
    ax2.set(xlabel='# Params', ylabel='Gap (%)', title='Gap vs # Parameters')
    ax2.axhline(0, color='gray', ls='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}parameter_analysis.png', dpi=150)
    plt.close()
    print("Saved: parameter_analysis.png")


def plot_param_impact(df):
    """Box plots for each parameter."""
    adr_df = df[df['mode'] == 'adr'] if 'mode' in df.columns else df
    if len(adr_df) < 2: return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, p in zip(axes, ['mass', 'damping', 'friction']):
        with_p = adr_df[adr_df['config'].apply(lambda x: x.get(p, False))]['transfer_gap']
        without_p = adr_df[adr_df['config'].apply(lambda x: not x.get(p, False))]['transfer_gap']
        
        if len(with_p) > 0 and len(without_p) > 0:
            ax.boxplot([without_p, with_p], labels=['Without', 'With'])
            ax.set(ylabel='Gap (%)', title=f'{p.capitalize()}')
            ax.axhline(0, color='gray', ls='--', alpha=0.5)
            ax.text(1, without_p.mean(), f'μ={without_p.mean():.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(2, with_p.mean(), f'μ={with_p.mean():.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Individual Parameter Impact (ADR only)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}individual_param_impact.png', dpi=150)
    plt.close()
    print("Saved: individual_param_impact.png")


def save_latex(df):
    """Generate LaTeX table."""
    df_sorted = df.sort_values('transfer_gap', ascending=False)
    lines = ["\\begin{tabular}{llcccc}", "\\toprule",
             "Config & Mode & Source & Target & Gap \\\\", "\\midrule"]
    for _, r in df_sorted.iterrows():
        lines.append(f"{r['config_name']} & {r.get('mode','adr').upper()} & "
                    f"${r['source_mean']:.0f} \\pm {r['source_std']:.0f}$ & "
                    f"${r['target_mean']:.0f} \\pm {r['target_std']:.0f}$ & "
                    f"${r['transfer_gap']:+.1f}\\%$ \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    
    with open(f'{OUTPUT_DIR}ablation_table.tex', 'w') as f:
        f.write('\n'.join(lines))
    print("Saved: ablation_table.tex")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_results()
    
    if len(df) == 0:
        print("No results. Run ablation study first.")
        exit(1)
    
    print(f"Generating plots from {len(df)} configs...")
    plot_transfer_gaps(df)
    plot_mode_comparison(df)
    plot_param_analysis(df)
    plot_param_impact(df)
    save_latex(df)
    print("\n✅ Done!")
