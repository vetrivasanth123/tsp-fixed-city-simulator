# Fixed-City TSP Simulator

A lightweight simulator for the **Traveling Salesman Problem (TSP)** with a fixed set of cities.

This project provides a clean foundation for studying sequential decision-making on a deterministic TSP instance. The initial implementation uses a small fixed-city example (e.g., 5 cities) while keeping the simulator architecture flexible enough to support different cost models, larger instances, and parameterized TSP problems in later stages.

## Project Scope

The project is developed incrementally:

### Phase 1 — Fixed-City TSP Simulator

* Represent a fixed set of cities and their pairwise distances/costs.
* Support Euclidean cost and custom cost matrices.
* Maintain the current tour state.
* Apply city-selection actions.
* Track visited and unvisited cities.
* Calculate and accumulate tour cost.
* Save and replay simulation trajectories.
* Provide static and animated route visualization.
* Validate simulator behavior with unit tests.

### Phase 2 — Gym/Gymnasium Environment

Convert the simulator into a Gym/Gymnasium-compatible environment with well-defined observation and action spaces.

### Phase 3 — PPO / Reinforcement Learning

Use the environment as the foundation for reinforcement-learning experiments, initially using PPO.

### Phase 4 — MTP Extension

Extend the fixed-city formulation toward the broader MTP formulation, including parameterized problem instances and additional decision-making components required by the major project.

The current repository intentionally implements **only Phase 1**.

## Design Goals

The simulator is intentionally lightweight.

It does **not** attempt to reproduce a general-purpose physics simulator or the full EvoGym framework. The implementation contains only the components required for the TSP task.

The architecture is designed around:

* fixed TSP instances,
* configurable edge costs,
* explicit simulator state,
* simple action execution,
* reproducible trajectories,
* testable components,
* minimal dependencies,
* and future extensibility.

## Initial Problem

For a set of \(N\) cities, the simulator maintains a tour beginning from a randomly selected starting city.

At each step, an action selects an unvisited city. The simulator:

1. validates the action,
2. moves to the selected city,
3. updates the visited-city state,
4. accumulates the corresponding edge cost,
5. determines whether the tour is complete.

When all cities have been visited, the simulator closes the tour by returning to the starting city.

The objective is to minimize the **total tour cost**.

## Cost Model

The simulator separates **geometric distance** from the **optimization cost**.

By default, the edge cost is the Euclidean distance between cities. A custom cost matrix can also be supplied for the same fixed coordinates.

```text
City Coordinates
       │
       ├──→ Euclidean Distance
       │
       └──→ Custom Cost Matrix
                    │
                    ▼
              instance.cost()
                    │
                    ▼
             TSP Simulator
```

The simulator uses:

```python
instance.cost(city_a, city_b)
```

as the main interface for edge costs.

The Euclidean distance matrix is retained separately as geometric information.

## Initial Instances

The repository currently contains two 5-city instances:

```text
instances/five_cities.json
instances/five_cities_custom_cost.json
```

`five_cities.json` uses Euclidean distance as the default cost.

`five_cities_custom_cost.json` uses an explicitly defined cost matrix while retaining the same city coordinates.

Example custom cost matrix:

```text
[[ 0. 10. 20. 15.  8.]
 [10.  0. 12. 18. 14.]
 [20. 12.  0.  9. 16.]
 [15. 18.  9.  0. 11.]
 [ 8. 14. 16. 11.  0.]]
```

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

## Components

### `tsp/instance.py`

Defines the TSP problem instance.

Responsibilities include:

* city coordinates,
* Euclidean distance matrix,
* configurable cost matrix,
* cost/distance access,
* instance loading and validation.

The starting city is selected and maintained by the simulator, not by the instance.

### `tsp/simulator.py`

Contains the core fixed-city TSP simulator.

Responsibilities include:

* simulator initialization and reset,
* current-city tracking,
* visited-city tracking,
* action execution,
* cost accumulation,
* tour completion,
* state retrieval.

This is the main component that will later support a Gym/Gymnasium wrapper.

### `tsp/utils.py`

Contains general-purpose utilities such as:

* tour validation,
* tour closing,
* tour-cost calculation,
* Euclidean distance-matrix construction.

### `tsp/visualization.py`

Provides lightweight visualization of:

* city locations,
* constructed routes,
* final tours,
* edge costs,
* total cost.

It also saves simulation trajectories and replays them through an animated visualization.

### `instances/`

Stores TSP problem data separately from the simulator implementation so that additional instances and cost models can be introduced without changing the simulator.

### `examples/run_fixed_city.py`

Demonstrates:

1. selecting a cost model,
2. loading the corresponding instance,
3. creating the simulator,
4. executing a complete tour,
5. reporting the result,
6. saving the trajectory.

### `examples/run_visualize.py`

Loads the saved simulation, restores the corresponding TSP instance and cost model, and replays the exact action sequence through the animated visualization.

### `tests/`

Contains unit tests for the instance and simulator.

The current test suite verifies instance loading, distance/cost behavior, state transitions, action validation, custom costs, and tour completion.

## Simulation State

The simulator maintains:

```text
start_city
current_city
tour
visited
available_actions
total_cost
done
```

The action space is currently represented simply as the set of unvisited city indices.

For example:

```text
Current city: 3
Available actions: [0, 1, 2, 4]
```

Selecting city `0` moves the simulator from city `3` to city `0` and adds:

```python
instance.cost(3, 0)
```

to the accumulated cost.

## Trajectory Persistence

Completed simulations can be saved as JSON.

Example:

```json
{
  "instance": "five_cities_custom_cost",
  "start_city": 3,
  "actions": [0, 2, 4, 1],
  "tour": [3, 0, 2, 4, 1, 3],
  "total_cost": 83.0
}
```

The saved instance name and action sequence allow the visualization to reproduce the same trajectory using the same cost model.

## Validation

The current implementation has been validated through:

* **25 passing unit tests**
* Euclidean-cost verification
* custom-cost verification
* invalid-action checks
* duplicate-city checks
* complete-tour checks
* trajectory persistence
* end-to-end custom-cost simulation and replay

A verified custom-cost run produced:

```text
Tour:
[3, 0, 2, 4, 1, 3]

Total cost:
83.0
```

The visualization used the same custom cost matrix and reproduced the saved trajectory with the corresponding edge-cost labels.

## Development Direction

The implementation will evolve without changing the basic separation between:

```text
TSP Instance
      ↓
TSP Simulator
      ↓
Gym/Gymnasium Environment
      ↓
PPO / RL
      ↓
MTP Extension
```

This separation allows the fixed-city simulator to remain simple while providing a stable foundation for later research experiments.

## Relationship to EvoGym

The repository is an independent TSP implementation.

The simulator architecture is **conceptually inspired by the environment-oriented design philosophy of EvoGym**, particularly the separation of simulation/environment dynamics from learning algorithms.

No EvoGym source code, soft-body physics, robot morphology, actuator mechanics, mass-spring dynamics, robot tasks, or optimization implementations are used in this project.

Any third-party code that is actually incorporated will be identified in:

```text
THIRD_PARTY_LICENSES.md
```

and credited according to its applicable license.

## License

This project is released under the MIT License. See `LICENSE` for details.
