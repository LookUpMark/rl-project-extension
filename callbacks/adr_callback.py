"""
ADR Callback for Stable Baselines3.

Monitors training performance and adjusts environment difficulty dynamically.
"""
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class ADRCallback(BaseCallback):
    """
    Callback per Automatic Domain Randomization.
    Monitora il reward medio e adatta la difficoltà dell'ambiente CustomHopper.
    """
    def __init__(self, check_freq: int, verbose: int = 1):
        """
        Inizializza la callback ADR.
        
        Args:
            check_freq (int): Frequenza (in step) di controllo performance.
            verbose (int): Livello di verbosità.
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        
        # SOGLIE DI PERFORMANCE
        # Hopper-v4 "risolto" è ~3000.
        # TODO: Calibrare queste soglie in base ai risultati del training base
        self.threshold_high = 2000  # Se reward > 2000, aumenta difficoltà
        self.threshold_low = 1000   # Se reward < 1000, diminuisci difficoltà

    def _on_step(self) -> bool:
        """
        Metodo chiamato ad ogni step del training.
        Gestisce la logica di update dell'ADR e il logging.
        
        Returns:
            bool: True per continuare il training, False per fermarlo.
        """
        # Eseguiamo il controllo solo ogni check_freq step
        if self.n_calls % self.check_freq == 0:
            
            # Recupera la storia dei reward dal Monitor wrapper
            ep_info_buffer = self.model.ep_info_buffer
            if len(ep_info_buffer) > 0:
                # Calcola media degli ultimi episodi
                mean_reward = np.mean([ep["r"] for ep in ep_info_buffer])
                
                # Accedi all'ambiente "nudo" (bypassando i wrapper SB3)
                # SB3 wrappa l'ambiente in DummyVecEnv -> Monitor -> CustomHopper
                env_unwrapped = self.training_env.envs[0].unwrapped
                
                # Verifica che l'ambiente supporti ADR
                if hasattr(env_unwrapped, 'update_adr'):
                    # Chiama l'update nell'environment
                    status, adr_stats = env_unwrapped.update_adr(
                        mean_reward, self.threshold_low, self.threshold_high
                    )
                    
                    # LOGGING SU TENSORBOARD (Cruciale per il report)
                    self.logger.record("adr/mean_reward", mean_reward)
                    self.logger.record("adr/mass_range_delta", adr_stats["mass_range"])
                    self.logger.record("adr/damping_range_delta", adr_stats["damping_range"])
                    self.logger.record("adr/friction_range_delta", adr_stats["friction_range"])
                    
                    # Console output se verbose
                    if self.verbose > 0 and status != "stable":
                        print(f"[ADR] Step {self.num_timesteps}: {status.upper()} bounds. "
                              f"R={mean_reward:.0f} | "
                              f"MassΔ={adr_stats['mass_range']:.2f} | "
                              f"FrictionΔ={adr_stats['friction_range']:.2f}")
                else:
                    # Environment non compatibile con ADR
                    if self.verbose > 0:
                        print("[ADR Warning] Environment does not support update_adr method.")

        return True
