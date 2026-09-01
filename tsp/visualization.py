"""
Visualization utilities for the fixed-city TSP simulator.

This module contains:
- static city plots,
- static completed-tour plots,
- animated simulator visualization.

The animation directly drives the simulator one action at a time.
Therefore, the displayed trajectory corresponds exactly to the
actions executed by the simulator.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .instance import TSPInstance
from .utils import tour_length


# ---------------------------------------------------------------------
# Static city visualization
# ---------------------------------------------------------------------

def plot_cities(
    instance: TSPInstance,
    ax: Optional[plt.Axes] = None,
    show_labels: bool = True,
    title: str = "TSP City Locations",
) -> plt.Axes:
    """
    Plot the fixed city locations.
    """

    if ax is None:
        _, ax = plt.subplots()

    coordinates = np.asarray(
        instance.coordinates,
        dtype=float,
    )

    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=100,
        zorder=3,
    )

    if show_labels:
        for city_index, (x, y) in enumerate(coordinates):
            ax.annotate(
                str(city_index),
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=10,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    return ax


# ---------------------------------------------------------------------
# Static tour visualization
# ---------------------------------------------------------------------

def plot_tour(
    instance: TSPInstance,
    tour: Sequence[int],
    ax: Optional[plt.Axes] = None,
    show_labels: bool = True,
    show_distance: bool = True,
    close_tour: bool = True,
    title: str = "TSP Tour",
) -> plt.Axes:
    """
    Plot a completed TSP tour.

    The supplied tour should contain every city exactly once.
    """

    if ax is None:
        _, ax = plt.subplots()

    tour = list(tour)

    _validate_tour(
        instance,
        tour,
    )

    coordinates = np.asarray(
        instance.coordinates,
        dtype=float,
    )

    # Plot cities.
    plot_cities(
        instance,
        ax=ax,
        show_labels=show_labels,
        title=title,
    )

    # Build route.
    route = tour.copy()

    if close_tour:
        route.append(tour[0])

    route_coordinates = coordinates[route]

    ax.plot(
        route_coordinates[:, 0],
        route_coordinates[:, 1],
        marker="o",
        linewidth=2.0,
        zorder=2,
    )

    if show_distance:
        distance = tour_length(
            tour,
            instance.distance_matrix,
        )

        ax.set_title(
            f"{title} — Distance: {distance:.4f}"
        )

    return ax


# ---------------------------------------------------------------------
# Simulator static visualization
# ---------------------------------------------------------------------

def plot_tour_from_simulator(
    simulator,
    ax: Optional[plt.Axes] = None,
    show_labels: bool = True,
    show_distance: bool = True,
    title: str = "TSP Tour",
) -> plt.Axes:
    """
    Plot the current tour stored by a TSPSimulator.
    """

    return plot_tour(
        instance=simulator.instance,
        tour=simulator.tour,
        ax=ax,
        show_labels=show_labels,
        show_distance=show_distance,
        close_tour=simulator.done,
        title=title,
    )


# ---------------------------------------------------------------------
# Animated simulation
# ---------------------------------------------------------------------

def animate_simulation(
    simulator,
    action_selector: Optional[
        Callable[[dict], int]
    ] = None,
    interval: int = 1200,
    title: str = "Fixed-City TSP Simulation",
) -> FuncAnimation:
    """
    Animate the TSP simulator while it is executing.

    IMPORTANT
    ---------
    The animation itself performs the simulation.

    Each animation frame:
        1. reads the current simulator state,
        2. determines the next action,
        3. executes simulator.step(action),
        4. updates the visualization.

    Therefore the animation is not generated from a previously
    completed tour. The displayed route is produced at the same
    time as the simulator actions.

    Parameters
    ----------
    simulator:
        TSPSimulator instance.

    action_selector:
        Function that receives the current simulator state and
        returns the next city to select.

        If None, the first available action is selected.

        Later this can be replaced with:

            action_selector = agent.select_action

    interval:
        Time between animation frames in milliseconds.

    title:
        Base animation title.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Matplotlib animation object.
    """

    instance = simulator.instance

    coordinates = np.asarray(
        instance.coordinates,
        dtype=float,
    )

    if action_selector is None:

        def action_selector(state: dict) -> int:
            return state["available_actions"][0]

    # --------------------------------------------------------------
    # Figure and axes
    # --------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    # --------------------------------------------------------------
    # City coordinates
    # --------------------------------------------------------------

    ax.set_xlim(
        coordinates[:, 0].min() - 0.8,
        coordinates[:, 0].max() + 0.8,
    )

    ax.set_ylim(
        coordinates[:, 1].min() - 0.8,
        coordinates[:, 1].max() + 0.8,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------------------
    # City markers
    # --------------------------------------------------------------

    city_scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=120,
        zorder=4,
    )

    # --------------------------------------------------------------
    # City labels
    # --------------------------------------------------------------

    for city_index, (x, y) in enumerate(coordinates):

        ax.annotate(
            str(city_index),
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )

    # --------------------------------------------------------------
    # Route line
    # --------------------------------------------------------------

    route_line, = ax.plot(
        [],
        [],
        linewidth=2.5,
        marker="o",
        zorder=2,
    )

    # --------------------------------------------------------------
    # Current-city marker
    # --------------------------------------------------------------

    current_marker, = ax.plot(
        [],
        [],
        marker="o",
        markersize=16,
        markerfacecolor="none",
        markeredgewidth=3,
        linestyle="None",
        zorder=6,
    )

    # --------------------------------------------------------------
    # Candidate-action markers
    # --------------------------------------------------------------

    candidate_scatter = ax.scatter(
        [],
        [],
        s=180,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        zorder=5,
    )

    # --------------------------------------------------------------
    # Text information
    # --------------------------------------------------------------

    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
        ),
        zorder=10,
    )

    # --------------------------------------------------------------
    # Animation state
    # --------------------------------------------------------------

    finished = False
    action_history: list[int] = []

    def update(frame: int):
        nonlocal finished

        # ----------------------------------------------------------
        # If simulation is already finished, just display final state.
        # ----------------------------------------------------------

        if finished:

            state = simulator.state()

            _draw_animation_state(
                simulator=simulator,
                state=state,
                coordinates=coordinates,
                route_line=route_line,
                current_marker=current_marker,
                candidate_scatter=candidate_scatter,
                city_scatter=city_scatter,
                info_text=info_text,
                ax=ax,
                action_history=action_history,
                final=True,
            )

            return (
                route_line,
                current_marker,
                candidate_scatter,
                city_scatter,
                info_text,
            )

        # ----------------------------------------------------------
        # Current simulator state BEFORE action
        # ----------------------------------------------------------

        state = simulator.state()

        # ----------------------------------------------------------
        # If all cities have been visited, close the tour.
        # ----------------------------------------------------------

        if not state["available_actions"]:

            simulator.close_tour()

            finished = True

            state = simulator.state()

            _draw_animation_state(
                simulator=simulator,
                state=state,
                coordinates=coordinates,
                route_line=route_line,
                current_marker=current_marker,
                candidate_scatter=candidate_scatter,
                city_scatter=city_scatter,
                info_text=info_text,
                ax=ax,
                action_history=action_history,
                final=True,
            )

            return (
                route_line,
                current_marker,
                candidate_scatter,
                city_scatter,
                info_text,
            )

        # ----------------------------------------------------------
        # Determine next action.
        # ----------------------------------------------------------

        selected_action = action_selector(state)

        if selected_action not in state["available_actions"]:
            raise ValueError(
                f"Action selector returned invalid action "
                f"{selected_action}. "
                f"Available actions: "
                f"{state['available_actions']}"
            )

        # ----------------------------------------------------------
        # Execute the SAME action that is being visualized.
        # ----------------------------------------------------------

        simulator.step(selected_action)

        action_history.append(selected_action)

        # ----------------------------------------------------------
        # Obtain updated state.
        # ----------------------------------------------------------

        state = simulator.state()

        # ----------------------------------------------------------
        # Draw updated simulation.
        # ----------------------------------------------------------

        _draw_animation_state(
            simulator=simulator,
            state=state,
            coordinates=coordinates,
            route_line=route_line,
            current_marker=current_marker,
            candidate_scatter=candidate_scatter,
            city_scatter=city_scatter,
            info_text=info_text,
            ax=ax,
            action_history=action_history,
            final=False,
        )

        return (
            route_line,
            current_marker,
            candidate_scatter,
            city_scatter,
            info_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=instance.num_cities + 2,
        interval=interval,
        repeat=False,
        blit=False,
    )

    # Keep a reference to the simulator so it is obvious that
    # the animation and simulator belong to the same execution.
    animation.simulator = simulator

    return animation


# ---------------------------------------------------------------------
# Animation drawing helper
# ---------------------------------------------------------------------

def _draw_animation_state(
    simulator,
    state: dict,
    coordinates: np.ndarray,
    route_line,
    current_marker,
    candidate_scatter,
    city_scatter,
    info_text,
    ax: plt.Axes,
    action_history: list[int],
    final: bool,
) -> None:
    """
    Draw one simulator state onto the animation.
    """

    tour = state["tour"]

    current_city = state["current_city"]

    available_actions = state["available_actions"]

    start_city = state["start_city"]

    # --------------------------------------------------------------
    # Draw route accumulated so far.
    # --------------------------------------------------------------

    route_coordinates = coordinates[tour]

    route_line.set_data(
        route_coordinates[:, 0],
        route_coordinates[:, 1],
    )

    # --------------------------------------------------------------
    # Highlight current city.
    # --------------------------------------------------------------

    current_marker.set_data(
        [coordinates[current_city, 0]],
        [coordinates[current_city, 1]],
    )

    # --------------------------------------------------------------
    # Highlight available actions.
    # --------------------------------------------------------------

    if available_actions:

        candidate_coordinates = coordinates[
            available_actions
        ]

        candidate_scatter.set_offsets(
            candidate_coordinates
        )

    else:

        candidate_scatter.set_offsets(
            np.empty((0, 2))
        )

    # --------------------------------------------------------------
    # Highlight starting city.
    # --------------------------------------------------------------

    # City scatter is kept neutral; the current marker identifies
    # the currently active city.
    city_scatter.set_sizes(
        np.full(
            len(coordinates),
            120.0,
        )
    )

    # --------------------------------------------------------------
    # Build information panel.
    # --------------------------------------------------------------

    if final:

        info_text.set_text(
            f"SIMULATION COMPLETE\n\n"
            f"Start city: {start_city}\n"
            f"Tour: {tour}\n"
            f"Return: {start_city}\n"
            f"Total distance: "
            f"{simulator.total_distance:.4f}"
        )

        ax.set_title(
            f"Fixed-City TSP — Complete Tour "
            f"({simulator.total_distance:.4f})"
        )

    else:

        selected_action = (
            action_history[-1]
            if action_history
            else None
        )

        info_text.set_text(
            f"Current city: {current_city}\n"
            f"Visited: {tour}\n"
            f"Available actions: {available_actions}\n"
            f"Selected action: {selected_action}\n"
            f"Distance: "
            f"{simulator.total_distance:.4f}"
        )

        ax.set_title(
            "Fixed-City TSP — Building Tour"
        )


# ---------------------------------------------------------------------
# Tour validation
# ---------------------------------------------------------------------

def _validate_tour(
    instance: TSPInstance,
    tour: Iterable[int],
) -> None:
    """
    Validate that a tour contains every city exactly once.
    """

    tour = list(tour)

    num_cities = instance.num_cities

    if len(tour) != num_cities:
        raise ValueError(
            f"Tour must contain exactly {num_cities} cities; "
            f"received {len(tour)}."
        )

    if sorted(tour) != list(range(num_cities)):
        raise ValueError(
            "Tour must contain every city index exactly once."
        )
