from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .instance import TSPInstance
from .simulator import TSPSimulator


class TSPEnv(gym.Env):
    """Gymnasium interface for TSP simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        source: TSPInstance | TSPSimulator,
        seed: int | None = None,
    ):
        super().__init__()

        if isinstance(source, TSPSimulator):
            self.simulator = source
            self.instance = source.instance
        else:
            self.instance = source
            self.simulator = TSPSimulator(source, seed=seed)

        n = self.instance.num_cities

        self.action_space = spaces.Discrete(n)
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

        if action not in self.simulator.available_actions():
            raise ValueError(
                f"Invalid action {action}. "
                f"Available actions: {self.simulator.available_actions()}"
            )

        previous = self.simulator.total_cost
        state = self.simulator.step(action)
        reward = -(self.simulator.total_cost - previous)

        terminated = False

        if not self.simulator.available_actions():
            previous = self.simulator.total_cost
            state = self.simulator.close_tour()
            reward -= self.simulator.total_cost - previous
            terminated = True

        return (
            self._observation(state),
            float(reward),
            terminated,
            False,
            self._info(state),
        )

    def _observation(self, state):
        mask = np.zeros(self.instance.num_cities, dtype=np.int8)
        mask[state["visited"]] = 1

        return {
            "current_city": int(state["current_city"]),
            "visited_mask": mask,
            "total_cost": np.array(
                [state["total_cost"]], dtype=np.float32
            ),
        }

    def _info(self, state):
        return {
            "tour": list(state["tour"]),
            "start_city": int(state["start_city"]),
            "current_city": int(state["current_city"]),
            "available_actions": list(state["available_actions"]),
            "total_cost": float(state["total_cost"]),
        }
