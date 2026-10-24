"""Table of post-burn-in move acceptance for saved baseline chain 1.

The state CSV does not record proposed moves. Re-run its fixed seed and check
the resulting states against the CSV before reporting the move outcomes.
"""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE.parent))

from examples.multiple_change_point_model.example_run.example_run import (  # noqa: E402
    initial_state,
)
from examples.multiple_change_point_model.multiple_change_point_model import (  # noqa: E402
    DATA_FILE, END_DAY, MultipleChangePointModel, load_disaster_days,
)
from examples.multiple_change_point_model.multiple_change_point_model_moves import (  # noqa: E402
    BirthMove, DeathMove, HeightMove, PositionMove,
)
from common import RUN_DIR, style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


MOVE_ORDER = ("birth", "death", "height", "position")


def check_saved_chain(states, chain_id, burn_in, total_steps):
    """Ensure the reproduced proposals belong to the saved state trace."""
    path = RUN_DIR / "example_run_base.csv"
    checked = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["chain_id"] != chain_id:
                continue
            iteration = int(row["iteration"])
            if iteration != burn_in + checked + 1 or iteration > total_steps:
                raise ValueError("Saved baseline iterations do not match the run")
            state = states[iteration]
            if (state.k != int(row["k"])
                    or not np.array_equal(state.change_points,
                                          json.loads(row["change_points"]))
                    or not np.array_equal(state.intensities,
                                          json.loads(row["intensities"]))):
                raise ValueError(f"Reproduced chain differs at iteration {iteration}")
            checked += 1
    if checked != total_steps - burn_in:
        raise ValueError("Saved baseline chain has the wrong number of states")


def move_counts():
    metadata = json.loads((RUN_DIR / "example_run_metadata.json").read_text())
    if hashlib.sha256(DATA_FILE.read_bytes()).hexdigest() != metadata["data_sha256"]:
        raise ValueError("The disaster data no longer match the saved experiment")

    settings = metadata["common_prior"]
    prior_lambda = metadata["baseline"]["lambda"]
    chain = metadata["baseline"]["chains"][0]
    total_steps = metadata["total_updates_per_chain"]
    burn_in = metadata["burn_in_updates"]
    days = load_disaster_days()
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
        initial_state=initial_state(chain["initial_k"], len(days) / END_DAY),
        seed=chain["seed"],
    )
    states, outcomes = model.run(total_steps)
    check_saved_chain(states, chain["chain_id"], burn_in, total_steps)
    retained_outcomes = outcomes[burn_in:]
    proposed = Counter(name for name, _ in retained_outcomes)
    accepted = Counter(name for name, was_accepted in retained_outcomes
                       if was_accepted)
    if sum(proposed.values()) != total_steps - burn_in:
        raise AssertionError("One move must be proposed at each update")
    return chain["chain_id"], burn_in, proposed, accepted


def main():
    chain_id, burn_in, proposed, accepted = move_counts()
    style()
    rows = [
        (name.title(), f"{proposed[name]:,}", f"{accepted[name]:,}",
         f"{accepted[name] / proposed[name]:.3f}" if proposed[name] else "--")
        for name in MOVE_ORDER
    ]
    total_proposed = sum(proposed.values())
    total_accepted = sum(accepted.values())
    rows.append(("Overall", f"{total_proposed:,}", f"{total_accepted:,}",
                 f"{total_accepted / total_proposed:.3f}"))

    figure, axis = plt.subplots(figsize=(8.0, 1.95))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=("Move type", "Proposed", "Accepted", "Acceptance rate"),
        colWidths=(0.29, 0.21, 0.21, 0.29),
        cellLoc="center", colLoc="center", bbox=(0.01, 0.02, 0.98, 0.96),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dde5")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e7edf3")
            cell.set_text_props(weight="bold")
        elif row == len(rows):
            cell.set_facecolor("#f1f5f8")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    output = HERE / "acceptance_rate.pdf"
    figure.savefig(output, pad_inches=0.02)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
