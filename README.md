# Fixed-City TSP Simulator

A lightweight **fixed-city Traveling Salesman Problem (TSP)** simulator and Gymnasium environment designed as a foundation for studying sequential decision-making and reinforcement learning.

The project is developed incrementally, keeping the **TSP instance, environment, simulator, and future learning algorithms conceptually separated**.

---

## Project Status

| Phase   | Description                  | Status              |
| ------- | ---------------------------- | ------------------- |
| Phase 1 | Fixed-City TSP Simulator     | ✅ Complete          |
| Phase 2 | Gymnasium Environment        | ✅ Complete & Frozen |
| Phase 3 | PPO / Reinforcement Learning | 🔄 Current          |
| Phase 4 | MTP Extension                | ⏳ Future            |

Phase 3 is the current development phase. Later MTP components are intentionally deferred.

---

## Architecture

The current architecture is:

```text
TSP Instance
      ↓
   TSP Env
      ↓
TSPSimulator
      ↓
Episode State
      ↓
Observation / Action
      ↓
State Transition
      ↓
Reward / Termination
```

### Responsibilities

**TSPInstance**

* Stores city coordinates and edge costs.
* Provides Euclidean distances and optional custom cost matrices.
* Does not maintain episode state.

**TSPEnv**

* Provides the Gymnasium-compatible RL interface.
* Owns one `TSPSimulator`.
* Converts simulator state into observations.
* Converts transition cost into reward.
* Handles actions, termination, and environment interaction.

**TSPSimulator**

* Maintains the actual TSP episode state.
* Tracks the current city, tour, visited cities, available actions, and accumulated cost.
* Executes sequential city transitions and tour closure.
* Does not perform optimization or learning.

---

# Phase 1 — Fixed-City TSP Simulator

Phase 1 established the standalone TSP simulation core.

Implemented:

* Fixed city coordinates
* JSON-based TSP instances
* Euclidean distance matrix
* General edge-cost abstraction
* Custom cost matrix support
* Random starting city
* Reproducible seed support
* Sequential city-selection actions
* Visited/unvisited city tracking
* Invalid-action validation
* Prevention of repeated city visits
* Tour closure
* Cost accumulation
* Trajectory persistence
* Exact trajectory replay
* Static and animated visualization
* Automated testing

The simulator **executes and evaluates a supplied tour; it does not optimize the tour**.

### Cost model

The optimization cost is accessed through:

```python
instance.cost(city_a, city_b)
```

The default cost is Euclidean distance, while custom edge-cost matrices are also supported.

This separation allows future applications to represent quantities such as travel time, monetary cost, fuel consumption, or energy.

### Structural subtour prevention

The sequential formulation prevents city revisits during route construction:

```text
Current Tour
     ↓
Available Actions
     ↓
Only Unvisited Cities
     ↓
Select Next City
     ↓
Update Tour
```

This is a **sequential structural constraint**, not a classical MILP subtour-elimination formulation. No MTZ, DFJ, flow-based, or cut-generation constraints are implemented.

The return to the starting city is performed separately during tour closure.

---

# Phase 2 — Gymnasium Environment

Phase 2 placed a Gymnasium-compatible environment interface around the existing simulator.

Implemented:

* `TSPEnv`
* Gymnasium action space
* Gymnasium observation space
* `reset()`
* `step()`
* reward formulation
* episode termination
* invalid-action handling
* available-action reporting
* seed reproducibility
* custom-cost compatibility
* simulator/environment state consistency
* persistence and replay compatibility
* environment-specific tests

### Action space

For `N` cities:

```text
0 ... N-1 = city-selection actions
N         = CLOSE action
```

Only unvisited cities are valid city-selection actions.

The environment exposes the current valid actions through:

```python
info["available_actions"]
```

### Observation

The environment returns:

```python
{
    "current_city": ...,
    "visited_mask": ...,
    "total_cost": ...
}
```

### Reward

For a transition from city `i` to city `j` with cost `c(i,j)`:

```text
reward = -c(i,j)
```

The CLOSE action receives the negative cost of the final return to the starting city.

Therefore:

```text
Maximize cumulative reward
        ⇔
Minimize total tour cost
```

### Termination

The episode terminates only after the tour has visited all cities and the `CLOSE` action completes the return to the starting city.

### Validation

Phase 2 was validated with:

```text
40 / 40 tests passed
```

The test suite covers environment behavior, simulator integration, action validation, rewards, termination, reproducibility, and custom costs.

The generic Gymnasium `check_env()` could not complete its sampled-action determinism check because the TSP environment has **state-dependent valid actions**: the declared action space contains all possible action IDs, while only unvisited cities are valid in a given state. The environment intentionally rejects invalid actions, and this behavior is covered by the project's own tests.

The `total_cost` observation currently uses an infinite upper bound, which produces a non-fatal Gymnasium warning.

Phase 2 is therefore considered **complete, validated, and frozen**.

---

# Phase 3 — PPO / Reinforcement Learning

Phase 3 uses the completed Gymnasium environment as the foundation for reinforcement-learning experiments.

The development sequence is:

```text
Understand PPO
      ↓
PPO pseudocode
      ↓
Observation / action representation
      ↓
Valid-action handling
      ↓
Policy and value networks
      ↓
Rollout collection
      ↓
Advantage estimation
      ↓
PPO objective / update
      ↓
Training
      ↓
Evaluation
      ↓
Interpretation
```

Initial evaluation will include simple non-learning baselines where appropriate, followed by PPO training and comparison using tour cost and reward.

The objective is to understand:

```text
State
  ↓
Action
  ↓
Transition
  ↓
Reward
  ↓
Learning
  ↓
Tour Quality
```

PPO is kept outside the simulator and environment logic.

---

# Phase 4 — MTP Extension

Phase 4 will extend the foundation toward the broader MTP formulation.

Potential future components include:

* parameterized problem instances
* larger problem sizes
* richer state representations
* additional decision variables
* optimization components
* co-optimization
* interaction between design and operation
* additional RL/DL methods

The exact MTP architecture will be determined from the actual MTP requirements and relevant literature.

The TSP project is a **foundation**, not a miniature implementation of the complete MTP.

---

## Repository Structure

```text
tsp-fixed-city-simulator/
│
├── README.md
├── LICENSE
├── THIRD_PARTY_LICENSES.md
├── requirements.txt
├── .simulation.json
│
├── tsp/
│   ├── __init__.py
│   ├── instance.py
│   ├── simulator.py
│   ├── env.py
│   ├── utils.py
│   └── visualization.py
│
├── instances/
│   ├── five_cities.json
│   └── five_cities_custom_cost.json
│
├── examples/
│   ├── run_fixed_city.py
│   └── run_visualize.py
│
└── tests/
    ├── test_instance.py
    └── test_simulator.py
```

---

## Initial Instances

The repository contains two 5-city instances:

```text
instances/five_cities.json
instances/five_cities_custom_cost.json
```

The first uses Euclidean distance as the cost model. The second uses an explicitly defined custom cost matrix with the same city coordinates.

---

## Design Principles

The project follows these principles:

1. **Simulator ≠ Environment ≠ RL Agent**
2. The simulator remains the state-transition engine.
3. The environment provides the Gymnasium interface.
4. PPO consumes the environment and is not embedded in the simulator.
5. Invalid TSP actions are explicitly rejected.
6. Working phases are not redesigned unnecessarily.
7. Each phase is validated before the next phase begins.
8. Later MTP components are not introduced prematurely.
9. Small, understandable implementations are preferred over unnecessary abstraction.

---

## Relationship to EvoGym

The project is an independent TSP implementation.

Its architecture was **conceptually inspired by the environment-oriented design philosophy of EvoGym**, particularly the separation of simulation/environment dynamics from learning algorithms.

No EvoGym source code, soft-body physics, robot morphology, actuator mechanics, mass-spring dynamics, robot tasks, or optimization implementations are used in this project.

The conceptual relationship is limited to architectural ideas such as:

```text
Environment
    ↓
Simulator
    ↓
State
    ↓
Action
    ↓
Transition
    ↓
Observation / Reward
```

Any third-party code actually incorporated into the project will be identified in:

```text
THIRD_PARTY_LICENSES.md
```

and credited according to its applicable license.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
