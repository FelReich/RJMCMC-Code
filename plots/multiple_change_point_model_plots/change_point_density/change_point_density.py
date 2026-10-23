"""Green-style conditional KDEs for each ordered change-point coordinate."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    END_DAY, first_baseline_chain, fixed_gaussian_kde, style,
)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BANDWIDTH_DAYS = 625.0
CURVES = {1: ("-", "#254f75"), 2: (":", "#b46028"),
          3: ("--", "#448360")}


def main():
    style()
    draws = first_baseline_chain()
    grid = np.linspace(0, END_DAY, 800)
    figure, axis = plt.subplots(figsize=(8.3, 4.8))
    for k, (linestyle, color) in CURVES.items():
        conditional = [draw.state.change_points for draw in draws if draw.state.k == k]
        if not conditional:
            raise ValueError(f"No retained states for k={k}")
        positions = np.stack(conditional)
        # Each ordered coordinate has its own conditional marginal from chain 1.
        for j in range(k):
            density = fixed_gaussian_kde(positions[:, j], grid, BANDWIDTH_DAYS)
            axis.plot(grid, density, linestyle=linestyle, color=color,
                      linewidth=1.5, label=f"k = {k}" if j == 0 else None)
    axis.set(xlabel="Change-point position (days)",
             ylabel="Conditional posterior density (day$^{-1}$)")
    axis.set_xlim(0, END_DAY)
    axis.grid(alpha=0.2)
    axis.legend(title="Number of change points")
    figure.tight_layout()
    figure.savefig(Path(__file__).with_suffix(".pdf"))


if __name__ == "__main__":
    main()
