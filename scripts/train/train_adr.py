"""ADR training script - Automatic Domain Randomization with adaptive ranges."""
import gymnasium as gym
import numpy as np
import torch
import random
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *
from callbacks.adr_callback import ADRCallback

SEED = 42


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(run_name="5M", total_timesteps=5_000_000):
    set_seed(SEED)
    
    log_dir = f"./logs/adr/run_{run_name}/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environments
    env_source = Monitor(gym.make('CustomHopper-source-v0', udr=False))
    env_source.reset(seed=SEED)
    env_target = gym.make('CustomHopper-target-v0', udr=False)
    env_target.reset(seed=SEED)

    print(f'=== ADR TRAINING {run_name} (Seed: {SEED}) ===')
    print(f'Timesteps: {total_timesteps:,}')

    # Train with ADR callback
    adr_callback = ADRCallback(check_freq=2048)
    model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log=log_dir, seed=SEED)
    model.learn(total_timesteps=total_timesteps, callback=adr_callback, progress_bar=True)
    
    # Save
    model.save(f"./logs/adr/ppo_hopper_adr_{run_name}")

    # Evaluate
    mean_src, std_src = evaluate_policy(model, env_source, n_eval_episodes=50)
    mean_tgt, std_tgt = evaluate_policy(model, env_target, n_eval_episodes=50)
    
    gap = (mean_tgt - mean_src) / mean_src * 100 if mean_src > 0 else 0
    final_range = env_source.unwrapped.adr_state.get('mass_delta', 0)
    
    print(f"\nSource: {mean_src:.2f} ± {std_src:.2f}")
    print(f"Target: {mean_tgt:.2f} ± {std_tgt:.2f}")
    print(f"Transfer gap: {gap:+.1f}%")
    print(f"Final ADR range: ±{final_range*100:.1f}%")

    env_source.close()
    env_target.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ADR Training')
    parser.add_argument('--run', type=str, default='5M', choices=['2_5M', '5M', '10M'])
    args = parser.parse_args()
    
    timesteps = {'2_5M': 2_500_000, '5M': 5_000_000, '10M': 10_000_000}
    main(run_name=args.run, total_timesteps=timesteps[args.run])
