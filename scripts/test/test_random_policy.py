"""Test random policy on CustomHopper - useful for environment debugging."""
import gymnasium as gym
from env.custom_hopper import *


def main():
    render = True
    n_episodes = 10

    env = gym.make('CustomHopper-source-v0', render_mode='human' if render else None)

    print('State space:', env.observation_space)
    print('Action space:', env.action_space)
    print('Masses:', env.unwrapped.get_parameters())
    
    if hasattr(env.unwrapped, 'adr_state'):
        print('ADR State:', env.unwrapped.adr_state)

    for ep in range(n_episodes):
        done = False
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        print(f"Episode {ep + 1}: Steps={steps}, Reward={episode_reward:.2f}")

    env.close()


if __name__ == '__main__':
    main()
