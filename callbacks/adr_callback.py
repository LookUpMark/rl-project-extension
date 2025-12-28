"""ADR Callback - monitors performance and adjusts environment difficulty."""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class ADRCallback(BaseCallback):
    """Callback for Automatic Domain Randomization."""
    
    def __init__(self, check_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.threshold_high = 1200  # Expand if reward >= this
        self.threshold_low = 600    # Contract if reward < this

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
            
        ep_info_buffer = self.model.ep_info_buffer
        if len(ep_info_buffer) == 0:
            return True
            
        mean_reward = np.mean([ep["r"] for ep in ep_info_buffer])
        env_unwrapped = self.training_env.envs[0].unwrapped
        
        if not hasattr(env_unwrapped, 'update_adr'):
            return True
            
        # Update ADR bounds
        status, adr_stats = env_unwrapped.update_adr(
            mean_reward, self.threshold_low, self.threshold_high
        )
        
        # Log to tensorboard
        self.logger.record("adr/mean_reward", mean_reward)
        self.logger.record("adr/mass_range_delta", adr_stats["mass_range"])
        self.logger.record("adr/damping_range_delta", adr_stats["damping_range"])
        self.logger.record("adr/friction_range_delta", adr_stats["friction_range"])
        
        if self.verbose > 0 and status != "stable":
            print(f"[ADR] Step {self.num_timesteps}: {status.upper()} | "
                  f"R={mean_reward:.0f} | Range=±{adr_stats['mass_range']*100:.0f}%")

        return True
