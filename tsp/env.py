from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .instance import TSPInstance
from .simulator import TSPSimulator


class TSPEnv(gym.Env):
    """Gymnasium environment for sequential fixed-city TSP construction."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance: TSPInstance,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.instance = instance
        self.seed_value = seed

        n = instance.num_cities

        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict({
            "current_city": spaces.Discrete(n),
            "visited_mask": spaces.MultiBinary(n),
            "total_cost": spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(1,),
                dtype=np.float32,
            ),
        })

        self.simulator = TSPSimulator(instance, seed=seed)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        if seed is not None:
            self.simulator = TSPSimulator(
                self.instance,
                seed=seed,
            )
        else:
            self.simulator.reset()

        state = self.simulator.state()
        return self._observation(state), self._info(state)

    def step(self, action: int):
        action = self._validate_action(action)

        if action not in self.simulator.available_actions():
            raise ValueError(
                f"Invalid action {action}. "
                f"Available actions: {self.simulator.available_actions()}"
            )

        previous_cost = self.simulator.total_cost
        state = self.simulator.step(action)

        reward = -(self.simulator.total_cost - previous_cost)
        terminated = False
        truncated = False

        if not self.simulator.available_actions():
            previous_cost = self.simulator.total_cost
            state = self.simulator.close_tour()
            reward -= self.simulator.total_cost - previous_cost
            terminated = True

        return (
            self._observation(state),
            float(reward),
            terminated,
            truncated,
            self._info(state),
        )

    def _observation(self, state: dict) -> dict:
        mask = np.zeros(
            self.instance.num_cities,
            dtype=np.int8,
        )
        mask[state["visited"]] = 1

        return {
            "current_city": int(state["current_city"]),
            "visited_mask": mask,
            "total_cost": np.array(
                [state["total_cost"]],
                dtype=np.float32,
            ),
        }

    def _info(self, state: dict) -> dict:
        return {
            "tour": list(state["tour"]),
            "start_city": int(state["start_city"]),
            "current_city": int(state["current_city"]),
            "available_actions": list(state["available_actions"]),
            "total_cost": float(state["total_cost"]),
        }

    def _validate_action(self, action: int) -> int:
        if isinstance(action, np.integer):
            action = int(action)

        if not isinstance(action, int):
            raise TypeError("Action must be an integer city index.")

        if not self.action_space.contains(action):
            raise ValueError(
                f"Action {action} is outside the valid city range."
            )

        return action
