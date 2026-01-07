"""Shared utilities for training scripts."""
import os
import sys
import random
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from env.custom_hopper import *
from callbacks.adr_callback import ADRCallback

SEED = 42
TIMESTEPS_DEFAULT = 2_500_000


def set_seed(seed=SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_envs(udr_source=False, seed=SEED):
    """Create source and target environments."""
    src = Monitor(gym.make('CustomHopper-source-v0', udr=udr_source))
    src.reset(seed=seed)
    tgt = gym.make('CustomHopper-target-v0', udr=False)
    tgt.reset(seed=seed)
    return src, tgt


def train_and_evaluate(env_src, env_tgt, log_dir, timesteps=TIMESTEPS_DEFAULT, 
                       callback=None, model_name="ppo_hopper"):
    """Train PPO model and evaluate on both environments."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Train
    model = PPO('MlpPolicy', env_src, verbose=1, tensorboard_log=log_dir, seed=SEED)
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    model.save(os.path.join(log_dir, model_name))
    
    # Evaluate on clean environments
    eval_src = Monitor(gym.make('CustomHopper-source-v0', udr=False))
    eval_tgt = gym.make('CustomHopper-target-v0', udr=False)
    
    src_mean, src_std = evaluate_policy(model, eval_src, n_eval_episodes=50)
    tgt_mean, tgt_std = evaluate_policy(model, eval_tgt, n_eval_episodes=50)
    
    gap = (tgt_mean - src_mean) / src_mean * 100 if src_mean > 0 else 0
    
    eval_src.close()
    eval_tgt.close()
    
    return {
        'source': (src_mean, src_std),
        'target': (tgt_mean, tgt_std),
        'gap': gap,
        'model': model
    }


def print_results(results, name=""):
    """Print evaluation results."""
    src_mean, src_std = results['source']
    tgt_mean, tgt_std = results['target']
    print(f"\n{'='*50}")
    print(f"Results: {name}")
    print(f"Source: {src_mean:.0f} ± {src_std:.0f}")
    print(f"Target: {tgt_mean:.0f} ± {tgt_std:.0f}")
    print(f"Transfer gap: {results['gap']:+.1f}%")
    print(f"{'='*50}")
