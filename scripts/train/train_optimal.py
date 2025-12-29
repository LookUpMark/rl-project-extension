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
    
    # Create environment
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
    final_adr = env_source.unwrapped.get_adr_info()
    
    print(f"\nSource: {mean_src:.2f} ± {std_src:.2f}")
    print(f"Target: {mean_tgt:.2f} ± {std_tgt:.2f}")
    print(f"Transfer gap: {gap:+.1f}%")
    print(f"Final ADR: {final_adr}")
    
    # Save results
    results = {
        'enabled_params': enabled_params,
        'source_mean': float(mean_src),
        'target_mean': float(mean_tgt),
        'transfer_gap': float(gap),
        'final_adr': final_adr
    }
    with open(f"{log_dir}results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    env_source.close()
    env_target.close()


if __name__ == '__main__':
    main()
