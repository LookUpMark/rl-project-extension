"""Baseline training script - no domain randomization."""
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

SEED = 42


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(SEED)
    
    log_dir = "./logs/baseline/"
    os.makedirs(log_dir, exist_ok=True)
    total_timesteps = 2_500_000

    # Create environments
    env_source = Monitor(gym.make('CustomHopper-source-v0', udr=False))
    env_source.reset(seed=SEED)
    env_target = gym.make('CustomHopper-target-v0', udr=False)
    env_target.reset(seed=SEED)

    print(f'=== BASELINE TRAINING (Seed: {SEED}) ===')
    print(f'Timesteps: {total_timesteps:,}')

    # Train
    model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log=log_dir, seed=SEED)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save
    model.save(os.path.join(log_dir, "ppo_hopper_baseline"))

    # Evaluate
    mean_src, std_src = evaluate_policy(model, env_source, n_eval_episodes=50)
    mean_tgt, std_tgt = evaluate_policy(model, env_target, n_eval_episodes=50)
    
    gap = (mean_tgt - mean_src) / mean_src * 100
    print(f"\nSource: {mean_src:.2f} ± {std_src:.2f}")
    print(f"Target: {mean_tgt:.2f} ± {std_tgt:.2f}")
    print(f"Transfer gap: {gap:+.1f}%")

    env_source.close()
    env_target.close()


if __name__ == "__main__":
    main()
