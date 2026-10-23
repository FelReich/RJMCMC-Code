"""Green-style conditional KDEs for each ordered segment intensity."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import first_baseline_chain, fixed_gaussian_kde, style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BANDWIDTH_PER_DAY = 0.0003
CURVES = {1: ("-", "#254f75"), 2: (":", "#b46028"),
          3: ("--", "#448360")}


def main():
    style()
    draws = first_baseline_chain()
    conditional = {
        k: np.stack([draw.state.intensities for draw in draws if draw.state.k == k])
        for k in CURVES
    }
    if any(len(values) == 0 for values in conditional.values()):
        raise ValueError("No retained states for one of k=1,2,3")
    largest = max(0.012, max(
        float(np.quantile(values, 0.999) + 3 * BANDWIDTH_PER_DAY)
        for values in conditional.values()
    ))
    grid = np.linspace(0, largest, 850)
    figure, axis = plt.subplots(figsize=(8.3, 4.8))
    for k, (linestyle, color) in CURVES.items():
        # Ordered segment heights stay separate, as in Green's Fig. 4.
        for j in range(k + 1):
            density = fixed_gaussian_kde(conditional[k][:, j], grid,
                                         BANDWIDTH_PER_DAY)
            axis.plot(grid, density, linestyle=linestyle, color=color,
                      linewidth=1.5, label=f"k = {k}" if j == 0 else None)
    axis.set(xlabel="Segment intensity (disasters per day)",
             ylabel="Conditional posterior density (days)",
             title="Segment intensities conditional on k")
    axis.set_xlim(0, largest)
    axis.grid(alpha=0.2)
    axis.legend(title="Number of change points")
    figure.tight_layout()
    figure.savefig(Path(__file__).with_suffix(".pdf"))


if __name__ == "__main__":
    main()
