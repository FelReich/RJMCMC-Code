"""Quick end-to-end run of the coal-disaster change-point sampler.

Run with: python examples/multiple_change_point_model/test.py
"""

import sys
from pathlib import Path

# Allow this script to be run directly from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from examples.multiple_change_point_model.multiple_change_point_model import (
    END_DAY,
    MultipleChangePointModel,
    load_disaster_days,
)
from examples.multiple_change_point_model.multiple_change_point_model_moves import (
    BirthMove,
    DeathMove,
    HeightMove,
    PositionMove,
)


POISSON_LAMBDA = 3.0
K_MAX = 30
ALPHA = 1.0
BETA = 200.0
N_STEPS = 44_000
BURN_IN = 4_000
SEED = 2026
TRACE_PDF = Path(__file__).with_name("test_trace.pdf")


def main():
    days = load_disaster_days()
    settings = dict(
        L=END_DAY,
        alpha=ALPHA,
        beta=BETA,
        param=POISSON_LAMBDA,
        kmax=K_MAX,
    )
    move_settings = dict(settings, days=days)
    moves = {
        "height": HeightMove(**move_settings),
        "position": PositionMove(**move_settings),
        "birth": BirthMove(**move_settings),
        "death": DeathMove(**move_settings),
    }
    model = MultipleChangePointModel(
        moves=moves,
        **settings,
        seed=SEED,
    )
    states, outcomes = model.run(N_STEPS)
    evaluation = model.evaluate(burn_in=BURN_IN, trace_path=TRACE_PDF)
    plt.close(evaluation["trace_plot"])
    visited_k = np.array([state.k for state in states])

    print(f"Disasters: {len(days)}; observation window: 0 to {END_DAY} days")
    print(f"Steps: {N_STEPS}; initial k: {states[0].k}; final k: {model.state.k}")
    print(f"Range of k visited: {visited_k.min()} to {visited_k.max()}")
    print(f"Acceptance after {BURN_IN} burn-in steps:")
    print(f"  overall  {evaluation['acceptance_rate']['overall']:.1%}")
    recent = outcomes[BURN_IN:]
    for name in moves:
        attempts = sum(move == name for move, _ in recent)
        accepted = sum(move == name and was_accepted for move, was_accepted in recent)
        rate = evaluation["acceptance_rate"]["by_move"][name]
        rate_text = f"{rate:.1%}" if rate is not None else "not proposed"
        print(f"  {name:8s} {accepted:4d} / {attempts:4d} ({rate_text})")

    mixing = evaluation["mixing"]
    lag1 = mixing["lag1_autocorrelation"]
    ess = mixing["approximate_ess"]
    lag1_text = f"{lag1:.3f}" if lag1 is not None else "undefined"
    ess_text = f"{ess:.0f}" if ess is not None else "undefined"
    print(f"Model-index lag-1 autocorrelation: {lag1_text}")
    print(f"Approximate single-chain ESS for k: {ess_text}")
    print(f"Model-switch rate: {mixing['model_switch_rate']:.1%}")
    frequencies = mixing["model_index_frequencies"]
    frequency_text = ", ".join(
        f"{k}: {frequency:.1%}" for k, frequency in frequencies.items()
    )
    print("Model-index frequencies:", frequency_text)
    print(f"Trace plot: {TRACE_PDF}")

    print(
        "Final change points (days):",
        np.array2string(model.state.change_points, precision=1),
    )
    print(
        "Final intensities (per day):",
        np.array2string(model.state.intensities, precision=5),
    )


if __name__ == "__main__":
    main()
