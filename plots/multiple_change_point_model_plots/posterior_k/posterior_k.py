"""Pooled posterior mass function of the number of change points."""

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import all_draws, baseline_chains, style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    style()
    draws = all_draws(baseline_chains())
    indices, counts = np.unique([draw.state.k for draw in draws], return_counts=True)
    probabilities = counts / counts.sum()
    mean_k = float(np.dot(indices, probabilities))
    mode_k = int(indices[np.argmax(counts)])
    path = Path(__file__)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("k", "count", "posterior_probability",
                         "posterior_mean_k", "posterior_mode_k"))
        for k, count, probability in zip(indices, counts, probabilities):
            writer.writerow((int(k), int(count), repr(float(probability)),
                             repr(mean_k), mode_k))

    figure, axis = plt.subplots(figsize=(7.3, 4.3))
    axis.bar(indices, probabilities, width=0.82, color="#41658a")
    axis.set(xlabel="Number of change points k", ylabel="Posterior probability",
             title="Posterior distribution of k")
    axis.set_xticks(indices)
    axis.set_ylim(0, max(probabilities) * 1.15)
    axis.text(0.98, 0.96, f"Mean = {mean_k:.2f}; mode = {mode_k}",
              transform=axis.transAxes, ha="right", va="top")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
