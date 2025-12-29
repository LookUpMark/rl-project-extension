"""Generate ablation study visualizations including UDR vs ADR comparison."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = "docs/evaluation/figures/"
RESULTS_DIR = "logs/ablation/"


def load_results():
    results = []
    if not os.path.exists(RESULTS_DIR):
        return pd.DataFrame()
    for config_dir in os.listdir(RESULTS_DIR):
        path = os.path.join(RESULTS_DIR, config_dir, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
    return pd.DataFrame(results)


def plot_transfer_gap_comparison(df):
    """Bar chart comparing transfer gaps across ALL configurations."""
    plt.figure(figsize=(14, 7))
    df_sorted = df.sort_values('transfer_gap', ascending=True)
    
    # Color by mode
    colors = []
    for _, row in df_sorted.iterrows():
        mode = row.get('mode', 'adr')
        if mode == 'baseline':
            colors.append('#1f77b4')  # Blue
        elif mode == 'udr':
            colors.append('#ff7f0e')  # Orange
        else:
            colors.append('#2ca02c')  # Green for ADR
    
    bars = plt.barh(df_sorted['config_name'], df_sorted['transfer_gap'], color=colors)
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Transfer Gap (%)', fontsize=12)
    plt.ylabel('Configuration', fontsize=12)
    plt.title('Ablation Study: Transfer Gap by Configuration', fontsize=14)
    
    # Add values on bars
    for bar, val in zip(bars, df_sorted['transfer_gap']):
        plt.text(val + 0.5 if val >= 0 else val - 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', va='center', ha='left' if val >= 0 else 'right', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='Baseline'),
                       Patch(facecolor='#ff7f0e', label='UDR'),
                       Patch(facecolor='#2ca02c', label='ADR')]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}ablation_transfer_gap.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR}ablation_transfer_gap.png")
    plt.close()


def plot_mode_comparison(df):
    """Compare Baseline vs UDR vs ADR (best ADR configuration)."""
    # Group by mode
    modes = ['baseline', 'udr']
    adr_configs = df[df['mode'] == 'adr']
    
    # Find best ADR config
    if len(adr_configs) > 0:
        best_adr = adr_configs.loc[adr_configs['transfer_gap'].idxmax()]
        best_adr_name = best_adr['config_name']
    else:
        best_adr = None
    
    # Prepare data
    data = []
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        if len(mode_df) > 0:
            row = mode_df.iloc[0]
            data.append({
                'mode': mode.upper(),
                'source_mean': row['source_mean'],
                'target_mean': row['target_mean'],
                'transfer_gap': row['transfer_gap']
            })
    
    if best_adr is not None:
        data.append({
            'mode': f'Best ADR ({best_adr_name})',
            'source_mean': best_adr['source_mean'],
            'target_mean': best_adr['target_mean'],
            'transfer_gap': best_adr['transfer_gap']
        })
    
    if len(data) == 0:
        print("No data for mode comparison")
        return
    
    result_df = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart: Source vs Target
    x = np.arange(len(result_df))
    width = 0.35
    ax1.bar(x - width/2, result_df['source_mean'], width, label='Source', color='#2ca02c')
    ax1.bar(x + width/2, result_df['target_mean'], width, label='Target', color='#d62728')
    ax1.set_xticks(x)
    ax1.set_xticklabels(result_df['mode'], rotation=15, ha='right')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Baseline vs UDR vs Best ADR: Performance')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Bar chart: Transfer Gap
    colors = ['#1f77b4' if 'BASELINE' in m else '#ff7f0e' if 'UDR' in m else '#2ca02c' 
              for m in result_df['mode']]
    ax2.bar(result_df['mode'], result_df['transfer_gap'], color=colors)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Transfer Gap (%)')
    ax2.set_title('Transfer Performance Comparison')
    ax2.set_xticklabels(result_df['mode'], rotation=15, ha='right')
    
    for i, (mode, gap) in enumerate(zip(result_df['mode'], result_df['transfer_gap'])):
        ax2.text(i, gap + 1, f'{gap:+.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}mode_comparison.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR}mode_comparison.png")
    plt.close()


def plot_parameter_contribution_heatmap(df):
    """Heatmap and scatter showing parameter analysis."""
    params = ['mass', 'damping', 'friction']
    
    # Extract config flags
    df = df.copy()
    for p in params:
        df[f'{p}_enabled'] = df['config'].apply(lambda x: x.get(p, False))
    
    df['n_params'] = df[[f'{p}_enabled' for p in params]].sum(axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Heatmap
    heatmap_data = df[[f'{p}_enabled' for p in params]].astype(int)
    heatmap_data.columns = params
    heatmap_data.index = df['config_name']
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', ax=ax1, cbar=False)
    ax1.set_title('Parameter Configuration Matrix')
    
    # Scatter with mode colors
    colors = df['mode'].apply(lambda x: '#1f77b4' if x == 'baseline' else '#ff7f0e' if x == 'udr' else '#2ca02c')
    ax2.scatter(df['n_params'], df['transfer_gap'], s=100, c=colors, edgecolors='black')
    for _, row in df.iterrows():
        ax2.annotate(row['config_name'], (row['n_params'], row['transfer_gap']), 
                    fontsize=8, ha='center', va='bottom')
    ax2.set_xlabel('Number of Randomized Parameters')
    ax2.set_ylabel('Transfer Gap (%)')
    ax2.set_title('Transfer Gap vs Number of Parameters')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}parameter_analysis.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR}parameter_analysis.png")
    plt.close()


def plot_individual_param_impact(df):
    """Box plots showing impact of each parameter (ADR configs only)."""
    # Filter to ADR configs only for fair comparison
    adr_df = df[df['mode'] == 'adr'] if 'mode' in df.columns else df
    
    if len(adr_df) < 2:
        print("Not enough ADR configs for individual param analysis")
        return
    
    params = ['mass', 'damping', 'friction']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, param in zip(axes, params):
        with_param = adr_df[adr_df['config'].apply(lambda x: x.get(param, False))]['transfer_gap']
        without_param = adr_df[adr_df['config'].apply(lambda x: not x.get(param, False))]['transfer_gap']
        
        if len(with_param) > 0 and len(without_param) > 0:
            bp = ax.boxplot([without_param, with_param], labels=['Without', 'With'])
            ax.set_ylabel('Transfer Gap (%)')
            ax.set_title(f'{param.capitalize()} Impact')
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            
            # Add means
            mean_without = without_param.mean()
            mean_with = with_param.mean()
            ax.text(1, mean_without, f'μ={mean_without:.1f}', ha='center', va='bottom', fontsize=9)
            ax.text(2, mean_with, f'μ={mean_with:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Individual Parameter Impact on Transfer (ADR Configs Only)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}individual_param_impact.png', dpi=150)
    print(f"Saved: {OUTPUT_DIR}individual_param_impact.png")
    plt.close()


def generate_latex_table(df):
    """Generate LaTeX table for paper."""
    # Sort by transfer gap
    df_sorted = df.sort_values('transfer_gap', ascending=False)
    
    latex = "\\begin{tabular}{llcccc}\n\\toprule\n"
    latex += "Config & Mode & Source & Target & Gap \\\\\n\\midrule\n"
    
    for _, row in df_sorted.iterrows():
        mode = row.get('mode', 'adr').upper()
        latex += f"{row['config_name']} & {mode} & "
        latex += f"${row['source_mean']:.0f} \\pm {row['source_std']:.0f}$ & "
        latex += f"${row['target_mean']:.0f} \\pm {row['target_std']:.0f}$ & "
        latex += f"${row['transfer_gap']:+.1f}\\%$ \\\\\n"
    
    latex += "\\bottomrule\n\\end{tabular}\n"
    
    with open(f'{OUTPUT_DIR}ablation_table.tex', 'w') as f:
        f.write(latex)
    print(f"Saved: {OUTPUT_DIR}ablation_table.tex")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_results()
    
    if len(df) == 0:
        print("No ablation results found. Run ablation experiments first.")
        exit(1)
    
    print(f"Generating ablation visualizations from {len(df)} configurations...")
    plot_transfer_gap_comparison(df)
    plot_mode_comparison(df)
    plot_parameter_contribution_heatmap(df)
    plot_individual_param_impact(df)
    generate_latex_table(df)
    print("\n✅ All visualizations generated!")
