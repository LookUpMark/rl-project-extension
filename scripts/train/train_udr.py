"""Training script for Hopper with Uniform Domain Randomization (UDR).

This script trains a PPO agent with static/uniform randomization ranges.
Used to compare with ADR (adaptive) approach.
"""
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Import custom modules
from env.custom_hopper import *


def main():
    """
    Training with Uniform Domain Randomization.
    
    UDR uses fixed randomization ranges (unlike ADR which adapts).
    Expected results:
        - Lower Source performance (due to randomization)
        - Better Target transfer (reduced reality gap)
    """
    # --- CONFIGURATION ---
    log_dir = "./logs/udr/"
    os.makedirs(log_dir, exist_ok=True)
    
    total_timesteps = 2_500_000  # 2.5M for comparison

    # --- ENVIRONMENT SETUP ---
    # Source: Training environment WITH Uniform Domain Randomization
    # udr=True enables static randomization of link masses
    env_source = Monitor(gym.make('CustomHopper-source-v0', udr=True))
    
    # Target: Evaluation environment (no randomization)
    env_target = gym.make('CustomHopper-target-v0', render_mode='human', udr=False)

    print('=== UDR TRAINING (Uniform Domain Randomization) ===')
    print('State space:', env_source.observation_space)
    print('Action space:', env_source.action_space)
    print('Dynamics parameters:', env_source.unwrapped.get_parameters())

    # --- MODEL SETUP ---
    model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log=log_dir)

    # --- TRAINING ---
    print("--- STARTING UDR TRAINING ---")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("--- TRAINING COMPLETE ---")

    # --- SAVE MODEL ---
    model_path = os.path.join(log_dir, "ppo_hopper_udr")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # --- EVALUATION ---
    print("Evaluating on Source...")
    mean_src, std_src = evaluate_policy(model, env_source, n_eval_episodes=50, render=False)
    
    print("Evaluating on Target...")
    mean_tgt, std_tgt = evaluate_policy(model, env_target, n_eval_episodes=50, render=True)
    
    print(f"Mean reward on source: {mean_src:.2f} +/- {std_src:.2f}")
    print(f"Mean reward on target: {mean_tgt:.2f} +/- {std_tgt:.2f}")
    print(f"Transfer Performance: {((mean_tgt - mean_src) / mean_src * 100):+.1f}%")

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    plt.bar(['Source (UDR Training)', 'Target (Transfer)'], [mean_src, mean_tgt], 
            yerr=[std_src, std_tgt], capsize=10, color=['#ff7f0e', '#d62728'])
    plt.ylabel('Mean Reward')
    plt.title('UDR: Source vs Target Performance')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('performance_comparison_udr.png')
    print("Plot saved as performance_comparison_udr.png")
    plt.show()

    env_source.close()
    env_target.close()


if __name__ == '__main__':
    main()
