"""Training script for Hopper WITHOUT Domain Randomization (Baseline).

This script trains a standard PPO agent without any randomization.
Used as a baseline to demonstrate the reality gap problem.
"""
import gymnasium as gym
import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Import custom modules
from env.custom_hopper import *


def main():
    """
    Baseline training without any domain randomization.
    
    Expected results:
        - High Source performance
        - Degraded Target performance (reality gap)
    """
    # --- CONFIGURATION ---
    log_dir = "./logs/baseline/"
    os.makedirs(log_dir, exist_ok=True)
    
    total_timesteps = 2_500_000  # 2.5M for comparison

    # --- ENVIRONMENT SETUP ---
    # Source: Training environment WITHOUT any randomization
    env_source = Monitor(gym.make('CustomHopper-source-v0', udr=False))
    
    # Target: Evaluation environment
    env_target = gym.make('CustomHopper-target-v0', render_mode='human', udr=False)

    print('=== BASELINE TRAINING (No Domain Randomization) ===')
    print('State space:', env_source.observation_space)
    print('Action space:', env_source.action_space)
    print('Dynamics parameters:', env_source.unwrapped.get_parameters())

    # --- MODEL SETUP ---
    model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log=log_dir)

    # --- TRAINING ---
    print("--- STARTING BASELINE TRAINING ---")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("--- TRAINING COMPLETE ---")

    # --- SAVE MODEL ---
    model_path = os.path.join(log_dir, "ppo_hopper_baseline")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # --- EVALUATION ---
    print("Evaluating on Source...")
    mean_src, std_src = evaluate_policy(model, env_source, n_eval_episodes=50, render=False)
    
    print("Evaluating on Target...")
    mean_tgt, std_tgt = evaluate_policy(model, env_target, n_eval_episodes=50, render=True)
    
    print(f"Mean reward on source: {mean_src:.2f} +/- {std_src:.2f}")
    print(f"Mean reward on target: {mean_tgt:.2f} +/- {std_tgt:.2f}")
    print(f"Reality Gap: {((mean_tgt - mean_src) / mean_src * 100):.1f}%")

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    plt.bar(['Source (Training)', 'Target (Transfer)'], [mean_src, mean_tgt], 
            yerr=[std_src, std_tgt], capsize=10, color=['#2ca02c', '#d62728'])
    plt.ylabel('Mean Reward')
    plt.title('Baseline (No DR): Source vs Target Performance')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('performance_comparison_baseline.png')
    print("Plot saved as performance_comparison_baseline.png")
    plt.show()

    env_source.close()
    env_target.close()


if __name__ == '__main__':
    main()
