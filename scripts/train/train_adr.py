"""Training script for Hopper with Automatic Domain Randomization (ADR).

This script implements the full training pipeline:
1. Environment setup (Source with ADR, Target for evaluation)
2. PPO training with ADR Callback
3. Tensorboard logging for monitoring ADR evolution
4. Final evaluation and comparison plots
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
from callbacks.adr_callback import ADRCallback


def main():
    """
    Main function for Sim-to-Real training with ADR.
    
    Pipeline:
        1. Initialize Source (training) and Target (evaluation) environments
        2. Configure ADR Callback with performance thresholds
        3. Train PPO with ADR-enabled callback
        4. Evaluate on both environments
        5. Generate comparison plots
    """
    # --- CONFIGURATION ---
    run_name = "5M"  # Change to "2_5M" for 2.5M run
    log_dir = f"./logs/adr/run_{run_name}/"
    os.makedirs(log_dir, exist_ok=True)
    
    total_timesteps = 5_000_000  # 5M for maximum ADR expansion
    
    # --- ENVIRONMENT SETUP ---
    # Source: Training environment wrapped with Monitor for callback access
    # Note: udr=False because we use ADR (adaptive) instead of UDR (static)
    env_source = Monitor(gym.make('CustomHopper-source-v0', udr=False))
    
    # Target: Evaluation environment (represents the "real" robot)
    env_target = gym.make('CustomHopper-target-v0', render_mode='human', udr=False)

    print('State space:', env_source.observation_space)
    print('Action space:', env_source.action_space)
    print('Initial dynamics parameters:', env_source.unwrapped.get_parameters())

    # --- ADR CALLBACK SETUP ---
    # check_freq=2048 aligns with PPO's default n_steps
    # TODO: Implement ADRCallback in callbacks/adr_callback.py
    adr_callback = ADRCallback(check_freq=2048)

    # --- MODEL SETUP ---
    # tensorboard_log enables visualization of ADR metrics
    model = PPO('MlpPolicy', env_source, verbose=1, tensorboard_log=log_dir)

    # --- TRAINING ---
    print("--- STARTING TRAINING WITH ADR ---")
    model.learn(total_timesteps=total_timesteps, callback=adr_callback, progress_bar=True)
    print("--- TRAINING COMPLETE ---")

    # --- SAVE MODEL ---
    model_path = f"./logs/adr/ppo_hopper_adr_{run_name}"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # --- EVALUATION ---
    print("Evaluating on Source...")
    mean_src, std_src = evaluate_policy(model, env_source, n_eval_episodes=50, render=False)
    
    print("Evaluating on Target...")
    mean_tgt, std_tgt = evaluate_policy(model, env_target, n_eval_episodes=50, render=True)
    
    print(f"Mean reward on source: {mean_src:.2f} +/- {std_src:.2f}")
    print(f"Mean reward on target: {mean_tgt:.2f} +/- {std_tgt:.2f}")

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    plt.bar(['Source (ADR Training)', 'Target (Unseen)'], [mean_src, mean_tgt], 
            yerr=[std_src, std_tgt], capsize=10, color=['#2ca02c', '#d62728'])
    plt.ylabel('Mean Reward')
    plt.title('ADR Robustness: Source vs Target Performance')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('performance_comparison_adr.png')
    print("Plot saved as performance_comparison_adr.png")
    plt.show()

    env_source.close()
    env_target.close()


if __name__ == '__main__':
    main()
