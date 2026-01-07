"""ADR Callback - adapts environment difficulty based on agent performance."""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ADRCallback(BaseCallback):
    """Monitors reward and expands/contracts ADR ranges."""
    
    def __init__(self, check_freq=2048, threshold_high=1200, threshold_low=600, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.threshold_high = threshold_high  # Expand if reward >= this
        self.threshold_low = threshold_low    # Contract if reward < this

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True
        
        if len(self.model.ep_info_buffer) == 0:
            return True
        
        mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
        env = self.training_env.envs[0].unwrapped
        
        if not hasattr(env, 'update_adr'):
            return True
        
        status, adr = env.update_adr(mean_reward, self.threshold_low, self.threshold_high)
        
        # Log to tensorboard
        self.logger.record("adr/mean_reward", mean_reward)
        self.logger.record("adr/mass_range", adr["mass_range"])
        self.logger.record("adr/damping_range", adr["damping_range"])
        self.logger.record("adr/friction_range", adr["friction_range"])
        
        if self.verbose and status != "stable":
            print(f"[ADR] {self.num_timesteps}: {status} | R={mean_reward:.0f} | ±{adr['mass_range']*100:.0f}%")
        
        return True
