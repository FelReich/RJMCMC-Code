"""Read saved MCMC draws and provide figure-only helpers; never run a sampler."""

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.multiple_change_point_model.multiple_change_point_model import (  # noqa: E402
    END_DAY, State,
)
from rjmcmc.rjmcmc import BaseRJMCMC  # noqa: E402


RUN_DIR = (PROJECT_ROOT / "examples" / "multiple_change_point_model"
           / "example_run")
BASE_CSV = RUN_DIR / "example_run_base.csv"
ALTERNATIVE_CSV = RUN_DIR / "example_run_different_prior.csv"


@dataclass(frozen=True)
class StoredDraw:
    iteration: int
    prior_lambda: float
    state: State
    H: float


def expected_disasters(state):
    bounds = np.concatenate(([0.0], state.change_points, [END_DAY]))
    return float(np.dot(state.intensities, np.diff(bounds)))


def load_chains(path):
    """Rebuild every state; check dimensions, iterations, and the saved H."""
    chains = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"chain_id", "iteration", "prior_lambda", "k",
                    "change_points", "intensities", "H"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Missing required columns in {path}")
        for row in reader:
            state = State(int(row["k"]), json.loads(row["change_points"]),
                          json.loads(row["intensities"]))
            iteration = int(row["iteration"])
            H = float(row["H"])
            prior_lambda = float(row["prior_lambda"])
            if not (0 <= state.k <= 30 and iteration >= 1
                    and np.isfinite(prior_lambda) and prior_lambda > 0
                    and np.all(state.change_points > 0)
                    and np.all(state.change_points < END_DAY)
                    and np.isfinite(H)
                    and math.isclose(H, expected_disasters(state),
                                     rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f"Invalid saved state or H at iteration {iteration}")
            state.H = H  # Cached scalar for repeated diagnostic checkpoints.
            draws = chains.setdefault(row["chain_id"], [])
            if draws and iteration != draws[-1].iteration + 1:
                raise ValueError(f"Nonconsecutive iterations in {row['chain_id']}")
            if draws and prior_lambda != draws[-1].prior_lambda:
                raise ValueError("Prior lambda changed within a chain")
            draws.append(StoredDraw(iteration, prior_lambda, state, H))
    if not chains or any(not draws for draws in chains.values()):
        raise ValueError(f"No saved draws in {path}")
    lengths = {len(draws) for draws in chains.values()}
    if len(lengths) != 1:
        raise ValueError("Saved chains have unequal lengths")
    start_indices = {draws[0].iteration for draws in chains.values()}
    if len(start_indices) != 1:
        raise ValueError("Saved chains have different retained iteration ranges")
    return chains


def baseline_chains():
    chains = load_chains(BASE_CSV)
    if list(chains) != [f"base_{i}" for i in range(1, 6)]:
        raise ValueError("Expected five ordered baseline chains")
    return chains


def first_baseline_chain():
    return baseline_chains()["base_1"]


def all_draws(chains):
    return [draw for chain in chains.values() for draw in chain]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })


def saved_trace_plot(draws, statistic=None, label=None):
    """Reuse BaseRJMCMC.trace_plot and put the real CSV iteration on its x-axis.

    The base method expects an initial state preceding its recorded updates.
    A duplicate first retained state is supplied only as that synthetic initial
    state and removed from the displayed line immediately afterward.
    """
    reader = BaseRJMCMC({"stored": object()}, {"stored": 1.0})
    reader.states = [draws[0].state] + [draw.state for draw in draws]
    reader.move_outcomes = [("stored", True)] * len(draws)
    figure = reader.trace_plot(statistic=statistic, label=label)
    axis = figure.axes[0]
    line = axis.lines[0]
    line.set_data([draw.iteration for draw in draws], line.get_ydata()[1:])
    axis.relim()
    axis.autoscale_view()
    axis.set_xlim(draws[0].iteration, draws[-1].iteration)
    figure.tight_layout()
    return figure


def fixed_gaussian_kde(samples, grid, bandwidth):
    """A normalized Gaussian mixture with the specified fixed SD."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or len(samples) == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("KDE requires nonempty finite samples")
    if bandwidth <= 0:
        raise ValueError("Bandwidth must be positive")
    density = np.zeros_like(grid, dtype=float)
    for start in range(0, len(samples), 256):
        distances = (grid[:, None] - samples[None, start:start + 256]) / bandwidth
        density += np.exp(-0.5 * distances**2).sum(axis=1)
    return density / (len(samples) * bandwidth * np.sqrt(2 * np.pi))
