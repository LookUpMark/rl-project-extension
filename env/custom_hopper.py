"""CustomHopper with ADR (Automatic Domain Randomization) support."""
import os
from typing import Dict, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


class CustomHopper(MujocoEnv, utils.EzPickle):
    """Hopper with mass, damping, and friction randomization."""
    
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 125}

    def __init__(self, xml_file="hopper.xml", frame_skip=4, forward_reward_weight=1.0,
                 ctrl_cost_weight=1e-3, healthy_reward=1.0, terminate_when_unhealthy=True,
                 healthy_z_range=(0.7, float("inf")), healthy_angle_range=(-0.2, 0.2),
                 reset_noise_scale=5e-3, domain=None, udr=False, **kwargs):
        
        utils.EzPickle.__init__(self, xml_file, frame_skip, forward_reward_weight,
            ctrl_cost_weight, healthy_reward, terminate_when_unhealthy, 
            healthy_z_range, healthy_angle_range, reset_noise_scale, domain, udr, **kwargs)

        self._udr = udr
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale

        if xml_file == "hopper.xml":
            xml_file = os.path.join(os.path.dirname(__file__), "assets/hopper.xml")

        MujocoEnv.__init__(self, xml_file, frame_skip, observation_space=None, **kwargs)
        
        obs_size = self.data.qpos.size + self.data.qvel.size - 1
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        # Save nominal values
        self.original_masses = np.copy(self.model.body_mass[1:])
        self.original_damping = np.copy(self.model.dof_damping)
        self.original_friction = np.copy(self.model.geom_friction)

        # Source domain: -1kg torso offset
        if domain == 'source':
            self.model.body_mass[1] -= 1.0

        # ADR state
        self.adr_state = {"mass_range": 0.0, "damping_range": 0.0, "friction_range": 0.0}
        self.adr_step_size = 0.05
        self.min_friction_floor = 0.3
        self.adr_enabled_params = {'mass': True, 'damping': True, 'friction': True}

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]
        return (np.all(np.abs(state) < 100) and 
                self._healthy_z_range[0] < z < self._healthy_z_range[1] and
                self._healthy_angle_range[0] < angle < self._healthy_angle_range[1])

    def _get_obs(self):
        pos = self.data.qpos.flatten()[1:]  # Skip x position
        vel = np.clip(self.data.qvel.flatten(), -10, 10)
        return np.concatenate([pos, vel])

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_velocity = (self.data.qpos[0] - x_before) / self.dt

        # Reward
        forward = self._forward_reward_weight * x_velocity
        healthy = self._healthy_reward if self.is_healthy else 0
        ctrl = self._ctrl_cost_weight * np.sum(np.square(action))
        reward = forward + healthy - ctrl

        terminated = not self.is_healthy and self._terminate_when_unhealthy
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_obs(), reward, terminated, False, {"x_velocity": x_velocity}

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-noise, noise, self.model.nv)
        self.set_state(qpos, qvel)

        # Apply randomization
        if self._udr or any(v > 0 for v in self.adr_state.values()):
            self.set_parameters(self.sample_parameters())
        
        return self._get_obs()

    def sample_parameters(self) -> Dict:
        """Sample physics parameters based on ADR state."""
        def sample_range(original, range_val, floor=None):
            low = original * (1 - range_val)
            high = original * (1 + range_val)
            if floor is not None:
                low = np.maximum(low, floor)
            return np.random.uniform(low, high)
        
        m_range = self.adr_state["mass_range"] if self.adr_enabled_params.get('mass') else 0
        d_range = self.adr_state["damping_range"] if self.adr_enabled_params.get('damping') else 0
        f_range = self.adr_state["friction_range"] if self.adr_enabled_params.get('friction') else 0
        
        return {
            "masses": [sample_range(m, m_range) for m in self.original_masses],
            "damping": [sample_range(d, d_range) for d in self.original_damping],
            "friction": [sample_range(f, f_range, self.min_friction_floor) for f in self.original_friction]
        }

    def set_parameters(self, params):
        """Apply physics parameters."""
        if isinstance(params, dict):
            self.model.body_mass[1:] = params["masses"]
            self.model.dof_damping[:] = params["damping"]
            self.model.geom_friction[:] = params["friction"]
        else:
            self.model.body_mass[1:] = params  # Legacy: mass only

    def get_parameters(self):
        return np.array(self.model.body_mass[1:])

    def update_adr(self, mean_reward: float, low_th: float, high_th: float) -> Tuple[str, Dict]:
        """Update ADR ranges based on performance."""
        status = "stable"
        param_map = {'mass_range': 'mass', 'damping_range': 'damping', 'friction_range': 'friction'}
        
        if mean_reward >= high_th:
            for k in self.adr_state:
                if self.adr_enabled_params.get(param_map[k]):
                    self.adr_state[k] = min(self.adr_state[k] + self.adr_step_size, 1.0)
            status = "expanded"
        elif mean_reward < low_th:
            for k in self.adr_state:
                if self.adr_enabled_params.get(param_map[k]):
                    self.adr_state[k] = max(self.adr_state[k] - self.adr_step_size, 0.0)
            status = "contracted"
        
        return status, self.adr_state.copy()

    def get_adr_info(self) -> Dict:
        return self.adr_state.copy()


# Register environments
gym.register(id="CustomHopper-v0", entry_point=f"{__name__}:CustomHopper", max_episode_steps=500)
gym.register(id="CustomHopper-source-v0", entry_point=f"{__name__}:CustomHopper", 
             max_episode_steps=500, kwargs={"domain": "source"})
gym.register(id="CustomHopper-target-v0", entry_point=f"{__name__}:CustomHopper", 
             max_episode_steps=500, kwargs={"domain": "target"})
