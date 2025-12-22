"""Test a random policy on the Gym Hopper environment.

Play around with this code to get familiar with the Hopper environment.

For example:
    - What happens if you don't reset the environment even after the episode is over?
    - When exactly is the episode over?
    - What is an action here?
    - How do ADR parameters affect the dynamics?

This script is useful for debugging and understanding the environment before training.
"""
import gymnasium as gym
from env.custom_hopper import *


def main():
    """
    Run random policy episodes on the Hopper environment.
    
    Useful for:
        - Verifying environment setup
        - Understanding state/action spaces
        - Testing ADR parameter changes
    """
    render = True
    n_episodes = 10  # Reduced for quick testing

    # --- ENVIRONMENT SETUP ---
    if render:
        env = gym.make('CustomHopper-source-v0', render_mode='human')
    else:
        env = gym.make('CustomHopper-source-v0')
    
    # Alternative: Test on target environment
    # env = gym.make('CustomHopper-target-v0', render_mode='human')

    print('State space:', env.observation_space)
    print('Action space:', env.action_space)
    print('Dynamics parameters (masses):', env.unwrapped.get_parameters())
    
    # --- ADR DEBUG INFO ---
    # Print ADR state if available
    if hasattr(env.unwrapped, 'adr_state'):
        print('ADR State:', env.unwrapped.adr_state)
    if hasattr(env.unwrapped, 'original_damping'):
        print('Original Damping:', env.unwrapped.original_damping)
    if hasattr(env.unwrapped, 'original_friction'):
        print('Original Friction shape:', env.unwrapped.original_friction.shape)

    # --- EPISODE LOOP ---
    for ep in range(n_episodes):
        done = False
        state, info = env.reset()
        episode_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()  # Random action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            if render:
                env.render()
        
        print(f"Episode {ep + 1}: Steps={steps}, Reward={episode_reward:.2f}")

    env.close()


if __name__ == '__main__':
    main()
