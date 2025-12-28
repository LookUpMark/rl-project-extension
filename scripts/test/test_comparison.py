"""Comparison test script - generates charts and evaluates all models."""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tensorboard.backend.event_processing import event_accumulator
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from env.custom_hopper import *

OUTPUT_DIR = "docs/evaluation/figures/"
LOGS_DIR = "logs/"
N_EVAL_EPISODES = 50


def get_tb_data(log_dir, tag):
    """Extract data from Tensorboard logs."""
    for root, _, files in os.walk(log_dir):
        for f in files:
            if 'events.out.tfevents' in f:
                ea = event_accumulator.EventAccumulator(os.path.join(root, f))
                ea.Reload()
                if tag in ea.Tags().get('scalars', []):
                    events = ea.Scalars(tag)
                    return np.array([e.step for e in events]), np.array([e.value for e in events])
    return None, None


def plot_training_curves():
    """Plot ADR range expansion and reward curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    adr_runs = [
        ('ADR 10M', f'{LOGS_DIR}adr/run_10M/', '#9467bd'),
        ('ADR 5M', f'{LOGS_DIR}adr/run_5M/', '#d62728'),
        ('ADR 2.5M', f'{LOGS_DIR}adr/run_2_5M/', '#2ca02c')
    ]
    
    for name, log_dir, color in adr_runs:
        if not os.path.exists(log_dir):
            continue
        
        steps, values = get_tb_data(log_dir, 'adr/mass_range_delta')
        if steps is not None:
            ax1.plot(steps/1e6, values*100, label=name, color=color, linewidth=2)
        
        steps, values = get_tb_data(log_dir, 'rollout/ep_rew_mean')
        if steps is not None:
            w = min(50, len(values)//10) or 1
            smoothed = np.convolve(values, np.ones(w)/w, mode='valid')
            ax2.plot(steps[:len(smoothed)]/1e6, smoothed, label=name, color=color, linewidth=1.5)
    
    ax1.set_xlabel('Timesteps (M)')
    ax1.set_ylabel('ADR Range (%)')
    ax1.set_title('ADR Range Expansion')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('Timesteps (M)')
    ax2.set_ylabel('Episode Reward')
    ax2.set_title('Training Performance')
    ax2.axhline(1200, color='green', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}training_curves.png")
    plt.close()


def plot_learning_curves():
    """Plot reward curves for all methods."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    all_runs = [
        ('Baseline', f'{LOGS_DIR}baseline/', '#1f77b4'),
        ('UDR', f'{LOGS_DIR}udr/', '#ff7f0e'),
        ('ADR 2.5M', f'{LOGS_DIR}adr/run_2_5M/', '#2ca02c'),
        ('ADR 5M', f'{LOGS_DIR}adr/run_5M/', '#d62728'),
        ('ADR 10M', f'{LOGS_DIR}adr/run_10M/', '#9467bd')
    ]
    
    for name, log_dir, color in all_runs:
        if not os.path.exists(log_dir):
            continue
        steps, values = get_tb_data(log_dir, 'rollout/ep_rew_mean')
        if steps is not None and len(steps) > 0:
            w = min(50, len(values)//10) or 1
            smoothed = np.convolve(values, np.ones(w)/w, mode='valid')
            ax.plot(steps[:len(smoothed)]/1e6, smoothed, label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Timesteps (M)')
    ax.set_ylabel('Episode Reward (smoothed)')
    ax.set_title('Learning Curves: All Methods')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}learning_curves.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}learning_curves.png")
    plt.close()


def evaluate_model_on_envs(model, use_randomized_source=False):
    """Evaluate model on source and target environments."""
    env_src = Monitor(gym.make('CustomHopper-source-v0', udr=use_randomized_source))
    env_tgt = Monitor(gym.make('CustomHopper-target-v0', udr=False))
    
    src_mean, src_std = evaluate_policy(model, env_src, n_eval_episodes=N_EVAL_EPISODES)
    tgt_mean, tgt_std = evaluate_policy(model, env_tgt, n_eval_episodes=N_EVAL_EPISODES)
    
    env_src.close()
    env_tgt.close()
    return {'src': (src_mean, src_std), 'tgt': (tgt_mean, tgt_std)}


def plot_final_performance():
    """Evaluate all models and generate comparison chart."""
    print("\n--- STANDARD EVALUATION (clean source) ---")
    
    models = {
        'Baseline': f'{LOGS_DIR}baseline/ppo_hopper_baseline.zip',
        'UDR': f'{LOGS_DIR}udr/ppo_hopper_udr.zip', 
        'ADR 2.5M': f'{LOGS_DIR}adr/ppo_hopper_adr_2_5M.zip',
        'ADR 5M': f'{LOGS_DIR}adr/ppo_hopper_adr_5M.zip',
        'ADR 10M': f'{LOGS_DIR}adr/ppo_hopper_adr_10M.zip'
    }
    
    results = {}
    for name, path in models.items():
        if os.path.exists(path):
            print(f"Evaluating {name}...")
            model = PPO.load(path)
            results[name] = evaluate_model_on_envs(model)
    
    if not results:
        print("No models found!")
        return results
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(results))
    w = 0.35
    names = list(results.keys())
    
    src_means = [results[n]['src'][0] for n in names]
    tgt_means = [results[n]['tgt'][0] for n in names]
    src_stds = [results[n]['src'][1] for n in names]
    tgt_stds = [results[n]['tgt'][1] for n in names]
    
    ax.bar(x - w/2, src_means, w, yerr=src_stds, label='Source', color='#2ca02c', capsize=4)
    ax.bar(x + w/2, tgt_means, w, yerr=tgt_stds, label='Target', color='#d62728', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Mean Reward')
    ax.set_title(f'Final Performance ({N_EVAL_EPISODES} episodes)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, (s, t) in enumerate(zip(src_means, tgt_means)):
        ax.text(i - w/2, s + 50, f'{s:.0f}', ha='center', fontsize=8)
        ax.text(i + w/2, t + 50, f'{t:.0f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}final_performance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}final_performance.png")
    
    # LaTeX table
    with open(f'{OUTPUT_DIR}results_table.tex', 'w') as f:
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Method & Source & Target & Gap & $n$ \\\\\n\\midrule\n")
        for n in names:
            s, ss = results[n]['src']
            t, ts = results[n]['tgt']
            gap = (t-s)/s*100 if s > 0 else 0
            f.write(f"{n} & ${s:.0f} \\pm {ss:.0f}$ & ${t:.0f} \\pm {ts:.0f}$ & ${gap:+.1f}\\%$ & {N_EVAL_EPISODES} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"Saved: {OUTPUT_DIR}results_table.tex")
    plt.close()
    return results


def plot_robustness_performance():
    """Evaluate ADR models with randomized source."""
    print("\n--- ROBUSTNESS EVALUATION (randomized source) ---")
    
    models = {
        'ADR 2.5M': f'{LOGS_DIR}adr/ppo_hopper_adr_2_5M.zip',
        'ADR 5M': f'{LOGS_DIR}adr/ppo_hopper_adr_5M.zip',
        'ADR 10M': f'{LOGS_DIR}adr/ppo_hopper_adr_10M.zip'
    }
    
    results = {}
    for name, path in models.items():
        if os.path.exists(path):
            print(f"Evaluating {name} (with randomized source)...")
            model = PPO.load(path)
            results[name] = evaluate_model_on_envs(model, use_randomized_source=True)
    
    if not results:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results))
    w = 0.35
    names = list(results.keys())
    
    src_means = [results[n]['src'][0] for n in names]
    tgt_means = [results[n]['tgt'][0] for n in names]
    src_stds = [results[n]['src'][1] for n in names]
    tgt_stds = [results[n]['tgt'][1] for n in names]
    
    ax.bar(x - w/2, src_means, w, yerr=src_stds, label='Source (Randomized)', color='#ff7f0e', capsize=4)
    ax.bar(x + w/2, tgt_means, w, yerr=tgt_stds, label='Target (Fixed)', color='#2ca02c', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Mean Reward')
    ax.set_title(f'ADR Robustness ({N_EVAL_EPISODES} episodes)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for i, (s, t) in enumerate(zip(src_means, tgt_means)):
        gap = (t-s)/s*100 if s > 0 else 0
        ax.text(i - w/2, s + 30, f'{s:.0f}', ha='center', fontsize=8)
        ax.text(i + w/2, t + 30, f'{t:.0f}', ha='center', fontsize=8)
        ax.text(i, max(s, t) + 100, f'Gap: {gap:+.0f}%', ha='center', fontsize=9, fontweight='bold',
               color='green' if gap > 0 else 'red')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}robustness_performance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}robustness_performance.png")
    
    print(f"\n--- ROBUSTNESS RESULTS ({N_EVAL_EPISODES} episodes) ---")
    for n in names:
        s, ss = results[n]['src']
        t, ts = results[n]['tgt']
        gap = (t-s)/s*100 if s > 0 else 0
        print(f"{n}: Source={s:.0f}±{ss:.0f}, Target={t:.0f}±{ts:.0f}, Gap={gap:+.1f}%")
    plt.close()


def print_summary(results):
    """Print final summary."""
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"\nEvaluation: {N_EVAL_EPISODES} episodes per model/environment")
    print("\n| Method | Source | Target | Transfer Gap |")
    print("|--------|--------|--------|--------------|")
    for name, data in results.items():
        s, ss = data['src']
        t, ts = data['tgt']
        gap = (t-s)/s*100 if s > 0 else 0
        print(f"| {name} | {s:.0f}±{ss:.0f} | {t:.0f}±{ts:.0f} | {gap:+.1f}% |")
    print("="*70)


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.chdir(os.path.join(os.path.dirname(__file__), '../..'))
    
    print("=== GENERATING COMPARISON CHARTS ===")
    print(f"Using {N_EVAL_EPISODES} evaluation episodes per model\n")
    
    plot_training_curves()
    plot_learning_curves()
    results = plot_final_performance()
    plot_robustness_performance()
    
    if results:
        print_summary(results)
    print("\n✅ Done!")
