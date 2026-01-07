"""Test random policy on CustomHopper - for environment debugging."""
import gymnasium as gym
from env.custom_hopper import *


def main(episodes=10, render=True):
    env = gym.make('CustomHopper-source-v0', render_mode='human' if render else None)
    
    print(f"Obs: {env.observation_space}, Act: {env.action_space}")
    print(f"Masses: {env.unwrapped.get_parameters()}")
    if hasattr(env.unwrapped, 'adr_state'):
        print(f"ADR: {env.unwrapped.adr_state}")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward, steps = 0, 0
        
        while True:
            state, reward, term, trunc, _ = env.step(env.action_space.sample())
            total_reward += reward
            steps += 1
            if term or trunc:
                break
        
        print(f"Ep {ep+1}: {steps} steps, reward={total_reward:.1f}")

    env.close()


if __name__ == '__main__':
    main()
