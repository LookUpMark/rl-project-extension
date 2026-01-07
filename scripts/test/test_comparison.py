"""Generate comparison charts and evaluate models."""
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

OUT = "docs/evaluation/figures/"
N_EVAL = 50

# Model paths
MODELS = {
    'Baseline': 'logs/baseline/ppo_hopper_baseline.zip',
    'UDR': 'logs/udr/ppo_hopper_udr.zip',
    'ADR 2.5M': 'logs/adr/ppo_hopper_adr_2_5M.zip',
    'ADR 5M': 'logs/adr/ppo_hopper_adr_5M.zip',
    'ADR 10M': 'logs/adr/ppo_hopper_adr_10M.zip'
}


def get_tb(log_dir, tag):
    """Get tensorboard data."""
    for root, _, files in os.walk(log_dir):
        for f in files:
            if 'events.out.tfevents' in f:
                ea = event_accumulator.EventAccumulator(os.path.join(root, f))
                ea.Reload()
                if tag in ea.Tags().get('scalars', []):
                    ev = ea.Scalars(tag)
                    return [e.step for e in ev], [e.value for e in ev]
    return [], []


def evaluate(model_path, rand_src=False):
    """Evaluate model on source/target."""
    model = PPO.load(model_path)
    src = Monitor(gym.make('CustomHopper-source-v0', udr=rand_src))
    tgt = gym.make('CustomHopper-target-v0')
    s_m, s_s = evaluate_policy(model, src, n_eval_episodes=N_EVAL)
    t_m, t_s = evaluate_policy(model, tgt, n_eval_episodes=N_EVAL)
    src.close(); tgt.close()
    return s_m, s_s, t_m, t_s


def plot_curves():
    """Training and learning curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for name, color in [('10M', '#9467bd'), ('5M', '#d62728'), ('2_5M', '#2ca02c')]:
        path = f'logs/adr/run_{name}/'
        if not os.path.exists(path): continue
        
        # ADR range
        steps, vals = get_tb(path, 'adr/mass_range')
        if steps:
            axes[0].plot(np.array(steps)/1e6, np.array(vals)*100, label=f'ADR {name}', color=color, lw=2)
        
        # Reward
        steps, vals = get_tb(path, 'rollout/ep_rew_mean')
        if steps:
            w = max(1, len(vals)//20)
            smooth = np.convolve(vals, np.ones(w)/w, 'valid')
            axes[1].plot(np.array(steps[:len(smooth)])/1e6, smooth, label=f'ADR {name}', color=color)
    
    axes[0].set(xlabel='Steps (M)', ylabel='Range (%)', title='ADR Expansion')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set(xlabel='Steps (M)', ylabel='Reward', title='Training')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}training_curves.png', dpi=150)
    plt.close()


def plot_performance():
    """Evaluate all models and make bar chart."""
    results = {}
    for name, path in MODELS.items():
        if os.path.exists(path):
            print(f"  {name}...", end=' ', flush=True)
            s_m, s_s, t_m, t_s = evaluate(path)
            results[name] = (s_m, s_s, t_m, t_s)
            print(f"src={s_m:.0f}, tgt={t_m:.0f}")
    
    if not results: return
    
    names = list(results.keys())
    src = [results[n][0] for n in names]
    tgt = [results[n][2] for n in names]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names)); w = 0.35
    ax.bar(x-w/2, src, w, label='Source', color='#2ca02c')
    ax.bar(x+w/2, tgt, w, label='Target', color='#d62728')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set(ylabel='Reward', title='Final Performance')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT}final_performance.png', dpi=150)
    plt.close()
    
    # Print summary
    print("\n| Model | Source | Target | Gap |")
    print("|-------|--------|--------|-----|")
    for n in names:
        s, _, t, _ = results[n]
        print(f"| {n} | {s:.0f} | {t:.0f} | {(t-s)/s*100:+.1f}% |")


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    os.chdir(os.path.join(os.path.dirname(__file__), '../..'))
    
    print("=== COMPARISON ===")
    print("\nTraining curves...")
    plot_curves()
    print(f"  Saved: training_curves.png")
    
    print("\nEvaluating models...")
    plot_performance()
    print(f"\n  Saved: final_performance.png")
    print("\n✅ Done!")
