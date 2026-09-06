from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .instance import TSPInstance
from .simulator import TSPSimulator


class TSPEnv(gym.Env):
    """Gymnasium environment for TSP."""

    metadata = {"render_modes": []}

    def __init__(self, instance: TSPInstance, seed: int | None = None):
        super().__init__()

        self.instance = instance
        self.simulator = TSPSimulator(instance, seed=seed)

        n = instance.num_cities

        self.close_action = n
        self.action_space = spaces.Discrete(n + 1)
        self.observation_space = spaces.Dict({
            "current_city": spaces.Discrete(n),
            "visited_mask": spaces.MultiBinary(n),
            "total_cost": spaces.Box(
                0.0, np.inf, shape=(1,), dtype=np.float32
            ),
        })

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.simulator._rng.seed(seed)

        state = self.simulator.reset()
        return self._observation(state), self._info(state)

    def step(self, action):
        action = int(action)
    
        if action == self.close_action:
            if self.simulator.available_actions():
                raise ValueError("Cannot close the tour while valid actions remain.")
    
            previous_cost = self.simulator.total_cost
            state = self.simulator.close_tour()
            reward = -(self.simulator.total_cost - previous_cost)
    
            return (
                self._observation(state),
                float(reward),
                True,
                False,
                self._info(state),
            )
    
        if action not in self.simulator.available_actions():
            raise ValueError(f"Invalid action: {action}")
    
        previous_cost = self.simulator.total_cost
        state = self.simulator.step(action)
        reward = -(self.simulator.total_cost - previous_cost)
    
        return (
            self._observation(state),
            float(reward),
            False,
            False,
            self._info(state),
        )

    def _observation(self, state):
        mask = np.zeros(self.instance.num_cities, dtype=np.int8)
        mask[state["visited"]] = 1

        return {
            "current_city": state["current_city"],
            "visited_mask": mask,
            "total_cost": np.array(
                [state["total_cost"]], dtype=np.float32
            ),
        }

    def _info(self, state):
        return {
            "tour": list(state["tour"]),
            "start_city": state["start_city"],
            "current_city": state["current_city"],
            "available_actions": list(state["available_actions"]),
            "total_cost": float(state["total_cost"]),
        }
