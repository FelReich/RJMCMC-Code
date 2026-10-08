"""Plot coal-mining disaster days and their cumulative counting process.

All event days are read from Code/data/coal_disasters.csv. Replacing that file
with a documented 192-event source will update the plot without code changes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_FILE = HERE.parents[1] / "data" / "coal_disasters.csv"
START_DAY = 0
END_DAY = 40907  # 1 January 1963, following 112 calendar years


def main() -> None:
    days = pd.read_csv(DATA_FILE)["day"].to_numpy(dtype=int)
    days.sort()
    if not len(days) or days[0] < START_DAY or days[-1] >= END_DAY:
        raise ValueError(f"Missing or out-of-range disaster days in {DATA_FILE}")

    # A right-continuous counting process: N(t) jumps by one at each event.
    count_x = np.r_[START_DAY, days, END_DAY]
    count_y = np.r_[0, np.arange(1, len(days) + 1), len(days)]

    fig, (ax_count, ax_events) = plt.subplots(
        2, 1, figsize=(8.0, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    ax_count.step(count_x, count_y, where="post", color="black",
                  linestyle=":", linewidth=1.5)
    ax_count.set_ylabel("Cumulative counts")
    ax_count.set_ylim(0, max(200, len(days) + 5))
    ax_count.grid(axis="y", color="0.88", linewidth=0.6)

    # Jitter changes only displayed vertical positions, never event days.
    rng = np.random.default_rng(2026)
    jitter = rng.uniform(-0.34, 0.34, size=len(days))
    ax_events.scatter(days, jitter, s=10, color="black", alpha=0.8,
                      linewidths=0)
    ax_events.set_ylim(-0.55, 0.55)
    ax_events.set_yticks([])
    ax_events.set_ylabel("Disasters")
    ax_events.set_xlabel("Time (days)")

    for ax in (ax_count, ax_events):
        ax.set_xlim(START_DAY, END_DAY)
        ax.spines[["top", "right"]].set_visible(False)
    ax_events.set_xticks(np.arange(0, 40001, 5000))

    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.13)
    fig.savefig(HERE / "figure_2-1.pdf")
    plt.close(fig)
    print(f"Plotted {len(days)} disasters from {DATA_FILE}")


if __name__ == "__main__":
    main()
