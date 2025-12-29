"""Ablation study training script - tests individual parameter contributions."""
import argparse
import gymnasium as gym
import numpy as np
import torch
import random
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from env.custom_hopper import *
from callbacks.adr_callback import ADRCallback

SEED = 42
TIMESTEPS = 2_500_000

# Ablation configurations - includes Baseline, UDR, and ADR variants (10 total)
ABLATION_CONFIGS = {
    # Baselines for comparison
    'baseline':  {'mass': False, 'damping': False, 'friction': False, 'mode': 'baseline'},
    'udr':       {'mass': True,  'damping': True,  'friction': True,  'mode': 'udr'},
    
    # ADR no params (control - ADR callback active but nothing enabled)
    'adr_none':  {'mass': False, 'damping': False, 'friction': False, 'mode': 'adr'},
    
    # ADR single parameter
    'adr_mass':  {'mass': True,  'damping': False, 'friction': False, 'mode': 'adr'},
    'adr_damp':  {'mass': False, 'damping': True,  'friction': False, 'mode': 'adr'},
    'adr_fric':  {'mass': False, 'damping': False, 'friction': True,  'mode': 'adr'},
    
    # ADR two parameters
    'adr_mass_damp': {'mass': True,  'damping': True,  'friction': False, 'mode': 'adr'},
    'adr_mass_fric': {'mass': True,  'damping': False, 'friction': True,  'mode': 'adr'},
    'adr_damp_fric': {'mass': False, 'damping': True,  'friction': True,  'mode': 'adr'},
    
    # ADR all parameters
    'adr_all':   {'mass': True,  'damping': True,  'friction': True,  'mode': 'adr'},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main(config_name):
    set_seed(SEED)
    
    config = ABLATION_CONFIGS[config_name]
    mode = config.get('mode', 'adr')
    log_dir = f"./logs/ablation/{config_name}/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create environment based on mode
    use_udr = (mode == 'udr')
    env_source = Monitor(gym.make('CustomHopper-source-v0', udr=use_udr))
    env_source.reset(seed=SEED)
    
    # For ADR mode, set enabled params
    if mode == 'adr':
        env_source.unwrapped.adr_enabled_params = {
            'mass': config['mass'],
            'damping': config['damping'],
            'friction': config['friction']
        }
    
    env_target = gym.make('CustomHopper-target-v0', udr=False)
    env_target.reset(seed=SEED)
    
    print(f'=== ABLATION: {config_name} (Seed: {SEED}) ===')
    print(f'Mode: {mode.upper()}')
    print(f'Params: mass={config["mass"]}, damp={config["damping"]}, fric={config["friction"]}')
    print(f'Timesteps: {TIMESTEPS:,}')
    
    # Train - use ADR callback only for ADR mode
    if mode == 'adr':
        adr_callback = ADRCallback(check_freq=2048)
        model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log=log_dir, seed=SEED)
        model.learn(total_timesteps=TIMESTEPS, callback=adr_callback, progress_bar=True)
    else:
        # Baseline or UDR - no callback needed
        model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log=log_dir, seed=SEED)
        model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    
    # Save model
    model.save(f"{log_dir}ppo_ablation_{config_name}")
    
    # Evaluate on clean environments
    eval_src = Monitor(gym.make('CustomHopper-source-v0', udr=False))
    eval_tgt = gym.make('CustomHopper-target-v0', udr=False)
    
    mean_src, std_src = evaluate_policy(model, eval_src, n_eval_episodes=50)
    mean_tgt, std_tgt = evaluate_policy(model, eval_tgt, n_eval_episodes=50)
    
    gap = (mean_tgt - mean_src) / mean_src * 100 if mean_src > 0 else 0
    final_adr = env_source.unwrapped.get_adr_info() if mode == 'adr' else {}
    
    # Save results
    results = {
        'config_name': config_name,
        'mode': mode,
        'config': config,
        'seed': SEED,
        'timesteps': TIMESTEPS,
        'source_mean': float(mean_src),
        'source_std': float(std_src),
        'target_mean': float(mean_tgt),
        'target_std': float(std_tgt),
        'transfer_gap': float(gap),
        'final_adr_state': final_adr,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{log_dir}results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSource: {mean_src:.2f} ± {std_src:.2f}")
    print(f"Target: {mean_tgt:.2f} ± {std_tgt:.2f}")
    print(f"Transfer gap: {gap:+.1f}%")
    if mode == 'adr':
        print(f"Final ADR: {final_adr}")
    
    env_source.close()
    env_target.close()
    eval_src.close()
    eval_tgt.close()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--config', type=str, required=True, 
                        choices=list(ABLATION_CONFIGS.keys()),
                        help='Ablation configuration to run')
    args = parser.parse_args()
    main(args.config)
