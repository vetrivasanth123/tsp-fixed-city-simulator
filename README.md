# Fixed-City TSP Simulator

A lightweight simulator for the **Traveling Salesman Problem (TSP)** with a fixed set of cities.

This project provides a clean foundation for studying sequential decision-making on a deterministic TSP instance. The initial implementation uses a small fixed-city example (e.g., 5 cities) while keeping the simulator architecture flexible enough to support larger and parameterized TSP instances in later stages.

## Project Scope

The project is developed incrementally:

### Phase 1 — Fixed-City TSP Simulator

* Represent a fixed set of cities and their pairwise distances.
* Maintain the current tour state.
* Apply city-selection actions.
* Track visited and unvisited cities.
* Calculate travel distance and tour completion.
* Provide basic route visualization.
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

* deterministic TSP instances,
* explicit simulator state,
* simple action execution,
* reproducible experiments,
* testable components,
* minimal dependencies,
* and future extensibility.

## Initial Problem

For a set of \(N\) cities, the simulator maintains a tour beginning from a designated starting city.

At each step, an action selects an unvisited city. The simulator:

1. validates the action,
2. moves to the selected city,
3. updates the visited-city state,
4. accumulates the travel distance,
5. determines whether the tour is complete.

When all cities have been visited, the simulator can close the tour by returning to the starting city.

The objective is to minimize the total tour distance.

## Initial Instance

The first example uses a fixed **5-city TSP instance** stored in:

```text
instances/five_cities.json
```

The instance format is intentionally separated from the simulator so that additional TSP instances can later be introduced without changing the simulator implementation.

## Repository Structure

```text
tsp-fixed-city-simulator/
│
├── README.md
├── LICENSE
├── THIRD_PARTY_LICENSES.md
├── requirements.txt
│
├── tsp/
│   ├── __init__.py
│   ├── instance.py
│   ├── simulator.py
│   ├── utils.py
│   └── visualization.py
│
├── instances/
│   └── five_cities.json
│
├── examples/
│   └── run_fixed_city.py
│
└── tests/
    ├── test_instance.py
    └── test_simulator.py
```

## Planned Components

### `tsp/instance.py`

Defines the TSP problem instance.

Responsibilities include:

* city representation,
* city coordinates,
* distance matrix,
* starting city,
* loading and validating instance data.

The instance is kept separate from the simulator so that the same simulator can later operate on different city configurations.

### `tsp/simulator.py`

Contains the core fixed-city TSP simulator.

Responsibilities include:

* simulator initialization,
* current-city tracking,
* visited-city tracking,
* action execution,
* distance accumulation,
* tour completion,
* tour retrieval.

This file contains the core task logic and is the main component that will later support a Gym/Gymnasium wrapper.

### `tsp/utils.py`

Contains small general-purpose utilities that do not belong to the simulator itself.

Examples may include:

* distance-matrix construction,
* validation helpers,
* reproducibility utilities.

### `tsp/visualization.py`

Provides lightweight visualization of:

* city locations,
* visited route,
* final tour,
* total distance.

The initial implementation is intended for simple Matplotlib-based use, including Google Colab.

### `instances/five_cities.json`

Stores the initial fixed 5-city problem instance.

Keeping the data outside the Python implementation makes it straightforward to introduce additional instances later.

### `examples/run_fixed_city.py`

A minimal executable example showing how to:

1. load the instance,
2. create the simulator,
3. execute a tour,
4. report the result,
5. visualize the route.

### `tests/`

Contains unit tests for the instance and simulator.

The tests should verify the simulator's mathematical and state-transition behavior before RL is introduced.

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

The project takes architectural inspiration from the separation of simulation logic and environment-facing interfaces used in EvoGym. It does **not** include EvoGym's physics simulator, robot morphology system, actuator mechanics, collision handling, mass-spring dynamics, or robot-specific code.

Any third-party code that is actually incorporated will be identified in:

```text
THIRD_PARTY_LICENSES.md
```

and credited according to its applicable license.

## License

This project is released under the MIT License. See `LICENSE` for details.
