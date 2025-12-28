"""CustomHopper environment with ADR (Automatic Domain Randomization) support."""
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class CustomHopper(MujocoEnv, utils.EzPickle):
    """Hopper environment with ADR support for mass, damping, and friction randomization."""
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
        "render_fps": 125,
    }

    def __init__(
        self,
        xml_file: str = "hopper.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        domain: Optional[str] = None,
        udr: bool = False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, xml_file, frame_skip, default_camera_config,
            forward_reward_weight, ctrl_cost_weight, healthy_reward,
            terminate_when_unhealthy, healthy_state_range, healthy_z_range,
            healthy_angle_range, reset_noise_scale,
            exclude_current_positions_from_observation, domain, udr, **kwargs,
        )

        self._udr = udr
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        if xml_file == "hopper.xml":
            xml_file = os.path.join(os.path.dirname(__file__), "assets/hopper.xml")

        MujocoEnv.__init__(
            self, xml_file, frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size - exclude_current_positions_from_observation
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

        # Save nominal values
        self.original_masses = np.copy(self.model.body_mass[1:])
        self.original_damping = np.copy(self.model.dof_damping)
        self.original_friction = np.copy(self.model.geom_friction)

        # Source domain has -1kg torso mass offset
        if domain == 'source':
            self.model.body_mass[1] -= 1.0

        # ADR state (ranges start at 0 = deterministic)
        self.adr_state = {"mass_range": 0.0, "damping_range": 0.0, "friction_range": 0.0}
        self.adr_step_size = 0.05
        self.min_friction_floor = 0.3

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]
        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range
        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle
        return all((healthy_state, healthy_z, healthy_angle))

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)
        if self._exclude_current_positions_from_observation:
            position = position[1:]
        return np.concatenate((position, velocity)).ravel()

    def step(self, action):
        x_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_after = self.data.qpos[0]
        x_velocity = (x_after - x_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info

    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        ctrl_cost = self.control_cost(action)
        reward = forward_reward + healthy_reward - ctrl_cost
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }
        return reward, reward_info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)

        # Apply UDR if enabled
        if self._udr:
            self.set_random_parameters()

        # Apply ADR if any range > 0
        if any([v > 0.0 for v in self.adr_state.values()]):
            self.set_parameters(self.sample_parameters())

        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    def set_random_parameters(self):
        """Legacy UDR method."""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses, damping, friction according to ADR state."""
        mass_range = self.adr_state["mass_range"]
        damping_range = self.adr_state["damping_range"]
        friction_range = self.adr_state["friction_range"]
        
        masses = [np.random.uniform(m * (1 - mass_range), m * (1 + mass_range)) 
                  for m in self.original_masses]
        damping = [np.random.uniform(d * (1 - damping_range), d * (1 + damping_range)) 
                   for d in self.original_damping]
        friction = [np.random.uniform(np.maximum(f * (1 - friction_range), self.min_friction_floor), 
                                       f * (1 + friction_range)) 
                    for f in self.original_friction]
        
        return {"masses": masses, "damping": damping, "friction": friction}

    def get_parameters(self):
        """Get current link masses."""
        return np.array(self.model.body_mass[1:])

    def set_parameters(self, task):
        """Set masses, damping, friction. Accepts dict (ADR) or list (legacy UDR)."""
        if isinstance(task, dict):
            self.model.body_mass[1:] = task["masses"]
            self.model.dof_damping[:] = task["damping"]
            self.model.geom_friction[:] = task["friction"]
        else:
            self.model.body_mass[1:] = task

    def update_adr(self, mean_reward: float, low_th: float, high_th: float) -> Tuple[str, Dict]:
        """Update ADR ranges based on performance. Called by ADRCallback."""
        status = "stable"
        if mean_reward >= high_th:
            for k in self.adr_state:
                self.adr_state[k] = min(self.adr_state[k] + self.adr_step_size, 1.0)
            status = "expanded"
        elif mean_reward < low_th:
            for k in self.adr_state:
                self.adr_state[k] = max(self.adr_state[k] - self.adr_step_size, 0.0)
            status = "contracted"
        return status, self.adr_state.copy()

    def get_adr_info(self) -> Dict:
        """Return current ADR state for logging."""
        return self.adr_state.copy()


# Register environments
gym.register(id="CustomHopper-v0", entry_point=f"{__name__}:CustomHopper", max_episode_steps=500)
gym.register(id="CustomHopper-source-v0", entry_point=f"{__name__}:CustomHopper", max_episode_steps=500, kwargs={"domain": "source"})
gym.register(id="CustomHopper-target-v0", entry_point=f"{__name__}:CustomHopper", max_episode_steps=500, kwargs={"domain": "target"})
