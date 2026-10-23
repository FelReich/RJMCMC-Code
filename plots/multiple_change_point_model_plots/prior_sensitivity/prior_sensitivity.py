"""Compare saved posterior draws under lambda=3 and one alternative prior."""

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    ALTERNATIVE_CSV, all_draws, first_baseline_chain, load_chains, style,
)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def summary(chains, label):
    draws = all_draws(chains)
    k = np.asarray([draw.state.k for draw in draws], dtype=int)
    H = np.asarray([draw.H for draw in draws], dtype=float)
    values, counts = np.unique(k, return_counts=True)
    probabilities = dict(zip(values.tolist(), (counts / len(k)).tolist()))
    lambdas = {draw.prior_lambda for draw in draws}
    if len(lambdas) != 1:
        raise ValueError("Mixed prior values within one experiment")
    lower, upper = np.quantile(H, [0.025, 0.975])
    row = {
        "prior": label,
        "lambda": next(iter(lambdas)),
        "n_chains": len(chains),
        "n_retained_draws": len(draws),
        "posterior_mean_H": float(H.mean()),
        "H_95_percent_lower": float(lower),
        "H_95_percent_upper": float(upper),
        "posterior_mean_k": float(k.mean()),
        "posterior_mode_k": int(values[np.argmax(counts)]),
        "convergence_note": ("Five-chain diagnostics reported separately"
                             if len(chains) > 1 else
                             "One chain; no multi-chain convergence diagnostic"),
    }
    return row, probabilities


def main():
    style()
    base_chains = {"base_1": first_baseline_chain()}
    alt_chains = load_chains(ALTERNATIVE_CSV)
    if list(alt_chains) != ["alternative_1"]:
        raise ValueError("Expected exactly one alternative-prior chain")
    base_row, base_mass = summary(base_chains, "baseline")
    alt_row, alt_mass = summary(alt_chains, "alternative")
    path = Path(__file__)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(base_row))
        writer.writeheader()
        writer.writerows((base_row, alt_row))

    indices = np.arange(max(max(base_mass), max(alt_mass)) + 1)
    width = 0.38
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(indices - width / 2, [base_mass.get(int(k), 0) for k in indices],
             width=width, color="#41658a", label=f"Baseline ($\\lambda$ = {base_row['lambda']:g})")
    axis.bar(indices + width / 2, [alt_mass.get(int(k), 0) for k in indices],
             width=width, color="#ba6b35", label=f"Alternative ($\\lambda$ = {alt_row['lambda']:g})")
    axis.set(xlabel="Number of change points k", ylabel="Posterior probability",
             title="Sensitivity of the posterior distribution of k")
    axis.set_xticks(indices)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    figure.text(0.5, 0.01,
                "One chain per prior; multi-chain convergence is not assessed here.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
