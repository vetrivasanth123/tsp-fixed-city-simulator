# Fixed-City TSP Simulator

A lightweight simulator for the **Traveling Salesman Problem (TSP)** with a fixed set of cities.

This project provides a clean foundation for studying sequential decision-making on a deterministic TSP instance. The initial implementation uses a small fixed-city example (5 cities) while keeping the simulator architecture flexible enough to support different cost models, larger instances, and parameterized TSP problems in later stages.

## Project Scope

The project is developed incrementally:

### Phase 1 — Fixed-City TSP Simulator

* Represent a fixed set of cities and their pairwise distances/costs.
* Support Euclidean cost and custom cost matrices.
* Maintain the current tour state.
* Apply city-selection actions.
* Track visited and unvisited cities.
* Prevent repeated city visits during sequential tour construction.
* Calculate and accumulate tour cost.
* Save and replay simulation trajectories.
* Provide static and animated route visualization.
* Validate simulator behavior with unit tests.

### Phase 2 — Gym/Gymnasium Environment

Convert the simulator into a Gym/Gymnasium-compatible environment with well-defined observation and action spaces, reward handling, termination, and invalid-action handling.

### Phase 3 — PPO / Reinforcement Learning

Use the environment as the foundation for reinforcement-learning experiments, initially using PPO.

### Phase 4 — MTP Extension

Extend the fixed-city formulation toward the broader MTP formulation, including parameterized problem instances and additional decision-making components required by the major project.

The current repository intentionally implements **only Phase 1**.

---

## Design Goals

The simulator is intentionally lightweight.

It does **not** attempt to reproduce a general-purpose physics simulator or the full EvoGym framework. The implementation contains only the components required for the TSP task.

The architecture is designed around:

* fixed TSP instances,
* configurable edge costs,
* explicit simulator state,
* sequential action execution,
* structural prevention of repeated-city visits,
* reproducible trajectories,
* testable components,
* minimal dependencies,
* and future extensibility.

---

## Initial Problem

For a set of \(N\) cities, the simulator maintains a tour beginning from a randomly selected starting city.

At each step, an action selects an **unvisited city**. The simulator:

1. validates the action,
2. moves to the selected city,
3. updates the visited-city state,
4. accumulates the corresponding edge cost,
5. updates the available actions,
6. continues until the route is completed.

When the tour is closed, the simulator returns to the starting city and adds the corresponding closing-edge cost.

The TSP formulation aims to minimize the **total tour cost**. The current simulator, however, **does not perform optimization**; it executes and evaluates the trajectory supplied by the action selector.

---

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

This separation allows the optimization cost to represent quantities other than physical distance, such as travel time, monetary cost, fuel consumption, energy, or another application-specific edge weight.

---

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
* available-action generation,
* action execution,
* duplicate-action validation,
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

The current example action selector chooses arbitrary valid actions for demonstration and is **not an optimization algorithm**.

### `examples/run_visualize.py`

Loads the saved simulation, restores the corresponding TSP instance and cost model, and replays the exact action sequence through the animated visualization.

### `tests/`

Contains unit tests for the instance and simulator.

The current test suite verifies instance loading, distance/cost behavior, state transitions, action validation, custom costs, repeated-city prevention, and tour completion.

---

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

The action space is currently represented as city indices, with only **unvisited cities exposed as valid actions**.

For example:

```text
Current city: 3
Visited: [3]
Available actions: [0, 1, 2, 4]
```

Selecting city `0` produces:

```text
Current city: 0
Visited: [3, 0]
Available actions: [1, 2, 4]
```

The selected edge contributes:

```python
instance.cost(3, 0)
```

to the accumulated cost.

---

## Structural Subtour Prevention

A valid TSP solution must form a **single Hamiltonian cycle**, rather than multiple disconnected cycles or premature cycles within the route.

The current sequential simulator prevents repeated-city visits structurally.

The mechanism is:

```text
Current tour
     │
     ▼
available_actions()
     │
     ▼
Remove already visited cities
     │
     ▼
Select an unvisited city
     │
     ▼
step(action)
```

For example, after:

```text
3 → 0
```

city `3` is already in the tour and is no longer available as a normal action.

Therefore, sequences such as:

```text
3 → 0 → 3
```

and:

```text
3 → 0 → 2 → 0
```

cannot be constructed through the normal valid-action mechanism.

The simulator also performs explicit validation inside `step()`. If a previously visited city is manually supplied, the action is rejected.

Conceptually:

```text
available_actions()
    ↓
only unvisited cities

step()
    ↓
reject already visited cities
```

This provides the simulator's **structural subtour-prevention mechanism**.

### Important distinction

This should **not** be confused with classical mathematical subtour-elimination constraints used in MILP formulations.

The current simulator does **not** implement:

* Miller-Tucker-Zemlin (MTZ) constraints,
* Dantzig-Fulkerson-Johnson (DFJ) constraints,
* flow-based subtour constraints,
* cut-generation methods,
* or other MILP subtour-elimination formulations.

Instead, subtours are prevented by the sequential action structure: cities cannot be revisited during route construction.

The return to the starting city is handled separately when the tour is closed.

This formulation is particularly suitable for the planned RL environment because the agent will construct the route sequentially rather than selecting all tour edges simultaneously.

---

## Tour Completion

Once the desired route has been constructed, the tour is closed using:

```python
simulator.close_tour()
```

For example:

```text
3 → 0 → 2 → 4 → 1
```

becomes:

```text
3 → 0 → 2 → 4 → 1 → 3
```

The final edge:

```text
1 → 3
```

is added to the accumulated cost.

The simulator maintains the constructed route as an open sequence until closure; the final repeated starting city represents the closing edge of the Hamiltonian cycle.

---

## Tour Cost

For a closed tour:

```text
v0 → v1 → v2 → ... → vn → v0
```

the total cost is:

```text
C =
c(v0,v1)
+ c(v1,v2)
+ ...
+ c(vn,v0)
```

where:

```text
c(i,j) = instance.cost(i,j)
```

The same formulation works for both Euclidean and custom cost matrices.

---

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

This also makes the simulation result independently inspectable after execution.

---

## Validation

The current implementation has been validated through:

* **25 passing unit tests**
* Euclidean-cost verification
* custom-cost verification
* invalid-action checks
* duplicate-city checks
* structural no-revisit/subtour checks
* complete-tour checks
* trajectory persistence
* end-to-end custom-cost simulation and replay
* cost-consistent visualization

A verified custom-cost run produced:

```text
Tour:
[3, 0, 2, 4, 1, 3]

Total cost:
83.0
```

For this trajectory:

```text
3 → 0 = 15
0 → 2 = 20
2 → 4 = 16
4 → 1 = 14
1 → 3 = 18
```

Therefore:

```text
15 + 20 + 16 + 14 + 18 = 83
```

The simulator reported:

```text
83.0
```

The visualization used the same custom cost matrix and reproduced the saved trajectory with the corresponding edge-cost labels.

---

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

The intended progression is:

```text
Phase 1 — Complete
Fixed-city simulator
      ↓
Cost abstraction
      ↓
Sequential transitions
      ↓
Subtour prevention
      ↓
Visualization
      ↓
Testing

Phase 2 — Next
Gym/Gymnasium wrapper
      ↓
Observation space
      ↓
Action space
      ↓
Invalid-action handling / masking
      ↓
Reward
      ↓
Termination

Phase 3
Baseline policies
      ↓
Random valid policy
      ↓
Simple heuristic
      ↓
Cost comparison

Phase 4
PPO / RL
      ↓
Training
      ↓
Evaluation

Phase 5
MTP extension
      ↓
Parameterized instances
      ↓
Larger experiments
      ↓
Additional decision-making components
```

The fixed-city simulator is therefore intended to remain a stable foundation while later layers are added around it.

---

## Relationship to EvoGym

The repository is an independent TSP implementation.

The simulator architecture is **conceptually inspired by the environment-oriented design philosophy of EvoGym**, particularly the separation of simulation/environment dynamics from learning algorithms.

No EvoGym source code, soft-body physics, robot morphology, actuator mechanics, mass-spring dynamics, robot tasks, or optimization implementations are used in this project.

The relevant architectural inspiration is limited to ideas such as:

* separating environment dynamics from learning,
* exposing state and actions,
* treating the environment as an independent experimental component,
* and supporting visualization/replay of experimental behavior.

Any third-party code that is actually incorporated will be identified in:

```text
THIRD_PARTY_LICENSES.md
```

and credited according to its applicable license.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
