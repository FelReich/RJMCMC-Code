"""Run and store the coal-disaster RJMCMC chains used by the figure scripts.

The CSVs contain post-update states only: iteration 4001 is the first draw
retained with the default 4000-update burn-in. JSON arrays round-trip binary64
values, so no MCMC simulation is needed to rebuild a sampled State.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from examples.multiple_change_point_model.multiple_change_point_model import (  # noqa: E402
    DATA_FILE, END_DAY, MultipleChangePointModel, State, load_disaster_days,
)
from examples.multiple_change_point_model.multiple_change_point_model_moves import (  # noqa: E402
    BirthMove, DeathMove, HeightMove, PositionMove,
)


OUTPUT_DIR = Path(__file__).resolve().parent
BASE_SEEDS = (20261022, 20261023, 20261024, 20261025, 20261026)
ALTERNATIVE_SEED = 20261027
FIELDS = ("chain_id", "iteration", "prior_lambda", "k", "change_points",
          "intensities", "H")


def expected_disasters(state):
    bounds = np.concatenate(([0.0], state.change_points, [END_DAY]))
    return float(np.dot(state.intensities, np.diff(bounds)))


def initial_state(k, empirical_rate):
    """Fixed, valid initial states with distinct dimensions across chains."""
    positions = np.linspace(0, END_DAY, k + 2)[1:-1]
    heights = empirical_rate * (1 + 0.1 * np.arange(k + 1))
    return State(k, positions, heights)


def run_chain(writer, chain_id, seed, initial_k, prior_lambda, settings,
              days, n_steps, burn_in):
    moves = {
        name: move(END_DAY, settings["alpha"], settings["beta"],
                   prior_lambda, settings["kmax"], days=days)
        for name, move in (
            ("height", HeightMove), ("position", PositionMove),
            ("birth", BirthMove), ("death", DeathMove),
        )
    }
    model = MultipleChangePointModel(
        moves=moves, alpha=settings["alpha"], beta=settings["beta"],
        param=prior_lambda, kmax=settings["kmax"], L=END_DAY,
        initial_state=initial_state(initial_k, len(days) / END_DAY), seed=seed,
    )
    states, _ = model.run(n_steps)
    written = 0
    for iteration in range(burn_in + 1, n_steps + 1):
        state = states[iteration]
        writer.writerow({
            "chain_id": chain_id,
            "iteration": iteration,
            "prior_lambda": repr(prior_lambda),
            "k": state.k,
            "change_points": json.dumps(state.change_points.tolist(),
                                        separators=(",", ":"), allow_nan=False),
            "intensities": json.dumps(state.intensities.tolist(),
                                      separators=(",", ":"), allow_nan=False),
            "H": repr(expected_disasters(state)),
        })
        written += 1
    if written != n_steps - burn_in:
        raise AssertionError("Incorrect number of retained states")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=44_000)
    parser.add_argument("--burn-in", type=int, default=4_000)
    parser.add_argument("--alternative-lambda", type=float, default=4.0)
    args = parser.parse_args()
    if args.steps <= args.burn_in or args.burn_in < 0:
        parser.error("Require steps > burn-in >= 0")
    if not np.isfinite(args.alternative_lambda) or args.alternative_lambda <= 0:
        parser.error("Alternative lambda must be finite and positive")

    settings = {"alpha": 1, "beta": 200, "kmax": 30}
    days = load_disaster_days()
    retained = args.steps - args.burn_in
    base_path = OUTPUT_DIR / "example_run_base.csv"
    alternative_path = OUTPUT_DIR / "example_run_different_prior.csv"

    with base_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        total = 0
        for number, seed in enumerate(BASE_SEEDS, start=1):
            chain_id = f"base_{number}"
            print(f"Running {chain_id} (seed {seed}, initial k={number - 1})", flush=True)
            total += run_chain(writer, chain_id, seed, number - 1, 3.0,
                               settings, days, args.steps, args.burn_in)
    if total != 5 * retained:
        raise AssertionError("Incorrect baseline row count")

    with alternative_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        print(f"Running alternative (seed {ALTERNATIVE_SEED}, initial k=2)", flush=True)
        alt_total = run_chain(writer, "alternative_1", ALTERNATIVE_SEED, 2,
                              args.alternative_lambda, settings, days,
                              args.steps, args.burn_in)
    if alt_total != retained:
        raise AssertionError("Incorrect alternative row count")

    metadata = {
        "data_file": str(DATA_FILE.relative_to(PROJECT_ROOT)),
        "data_sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "observation_end_day": END_DAY,
        "total_updates_per_chain": args.steps,
        "burn_in_updates": args.burn_in,
        "retained_states_per_chain": retained,
        "common_prior": settings,
        "baseline": {"lambda": 3.0, "chains": [
            {"chain_id": f"base_{i}", "seed": seed, "initial_k": i - 1}
            for i, seed in enumerate(BASE_SEEDS, start=1)
        ]},
        "alternative": {"lambda": args.alternative_lambda,
                        "chain_id": "alternative_1", "seed": ALTERNATIVE_SEED,
                        "initial_k": 2},
        "csv_columns": list(FIELDS),
    }
    (OUTPUT_DIR / "example_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {total:,} baseline and {alt_total:,} alternative states")


if __name__ == "__main__":
    main()
