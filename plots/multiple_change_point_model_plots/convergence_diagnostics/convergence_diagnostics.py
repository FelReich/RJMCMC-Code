"""Brooks-Giudici variance comparisons from the five saved baseline chains."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BaseRJMCMC, baseline_chains, style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DEFAULT_CHECKPOINTS = (1000, 2000, 4000, 6000, 8000, 12000, 16000,
                       20000, 24000, 30000, 36000, 40000)
COMPARISONS = (
    ("overall", "V_hat", "W_c", "V_hat_over_W_c"),
    ("within_model", "W_m", "W_m_W_c", "W_m_over_W_m_W_c"),
    ("between_model", "B_m", "B_m_W_c", "B_m_over_B_m_W_c"),
)


def counts_and_eligible(chains, n):
    counts = {chain_id: Counter(draw.state.k for draw in draws[:n])
              for chain_id, draws in chains.items()}
    eligible = sorted(set.intersection(*(
        {k for k, count in cell.items() if count >= 2}
        for cell in counts.values()
    )))
    return counts, eligible


def histories(chains, n):
    # The first retained draw is a synthetic initial state, sliced away by
    # two_way_anova(burn_in=0); all n CSV draws are therefore used exactly once.
    return [[draws[0].state] + [draw.state for draw in draws[:n]]
            for draws in chains.values()]


def h_statistic(state):
    return state.H


def k_statistic(state):
    return state.k


def final_rows(result, statistic, subset, all_models, coverage):
    rows = []
    pairs = COMPARISONS if statistic == "H" else COMPARISONS[:1]
    for comparison, numerator, denominator, ratio_name in pairs:
        ratio = result["ratios"][ratio_name]
        rows.append({
            "statistic": statistic,
            "model_subset": subset,
            "comparison": comparison,
            "numerator_name": numerator,
            "numerator": result[numerator],
            "denominator_name": denominator,
            "denominator": result[denominator],
            "variance_ratio": ratio,
            "sqrt_variance_ratio": (float(np.sqrt(ratio)) if ratio is not None
                                    and np.isfinite(ratio) and ratio >= 0 else ""),
            "selected_models": json.dumps(result["selected_models"]),
            "excluded_observed_models": json.dumps(sorted(
                set(all_models) - set(result["selected_models"]))),
            "visit_counts": json.dumps(result["visit_counts"], sort_keys=True),
            "retained_draws_per_chain": result["retained_draws_per_chain"],
            "selected_draw_coverage": coverage,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=int, nargs="*",
                        default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--reference-checkpoint", type=int, default=4000,
                        help="Select a fixed model set from this prefix")
    args = parser.parse_args()
    style()
    chains = baseline_chains()
    n_total = len(next(iter(chains.values())))
    if not 1 <= args.reference_checkpoint <= n_total:
        parser.error("Reference checkpoint must be within the retained chain")
    checkpoints = sorted(set(args.checkpoints) | {args.reference_checkpoint, n_total})
    if checkpoints[0] < 1 or checkpoints[-1] > n_total:
        parser.error("Every checkpoint must be within the retained chain")

    _, fixed_models = counts_and_eligible(chains, args.reference_checkpoint)
    _, all_final_eligible = counts_and_eligible(chains, n_total)
    if len(fixed_models) < 2 or len(all_final_eligible) < 2:
        raise ValueError("At least two models need two visits in every chain")
    all_final_observed = sorted({draw.state.k for draws in chains.values()
                                 for draw in draws})
    checkpoints_rows = []
    plotted = []
    for n in checkpoints:
        counts, eligible = counts_and_eligible(chains, n)
        missing = sorted(set(fixed_models) - set(eligible))
        row = {
            "retained_iterations_per_chain": n,
            "eligible_models_at_checkpoint": json.dumps(eligible),
            "fixed_selected_models": json.dumps(fixed_models),
            "excluded_observed_models": json.dumps(sorted(
                set(k for cell in counts.values() for k in cell) - set(fixed_models))),
            "visit_counts": json.dumps({i: dict(sorted(cell.items()))
                                        for i, cell in counts.items()}, sort_keys=True),
            "status": "skipped: fixed models not yet eligible" if missing else "included",
            "V_hat": "", "W_c": "", "W_m": "", "W_m_W_c": "",
            "B_m": "", "B_m_W_c": "",
            "V_hat_over_W_c": "", "W_m_over_W_m_W_c": "",
            "B_m_over_B_m_W_c": "",
        }
        if not missing:
            result = BaseRJMCMC.two_way_anova(
                histories(chains, n), h_statistic, models=fixed_models
            )
            for key in ("V_hat", "W_c", "W_m", "W_m_W_c", "B_m", "B_m_W_c"):
                row[key] = result[key]
            row.update(result["ratios"])
            plotted.append((n, result))
        checkpoints_rows.append(row)
    if len(plotted) < 2:
        raise ValueError("Need at least two eligible checkpoints for a progression")

    final_histories = histories(chains, n_total)
    all_draws_n = len(chains) * n_total
    def coverage(models):
        return sum(draw.state.k in models for draws in chains.values()
                   for draw in draws) / all_draws_n

    fixed_h = plotted[-1][1]
    fixed_k = BaseRJMCMC.two_way_anova(
        final_histories, k_statistic, models=fixed_models
    )
    full_h = BaseRJMCMC.two_way_anova(
        final_histories, h_statistic, models=all_final_eligible
    )
    full_k = BaseRJMCMC.two_way_anova(
        final_histories, k_statistic, models=all_final_eligible
    )
    rows = []
    for subset, models, h_result, k_result in (
        ("fixed_progression", fixed_models, fixed_h, fixed_k),
        ("all_final_eligible", all_final_eligible, full_h, full_k),
    ):
        rows.extend(final_rows(h_result, "H", subset, all_final_observed,
                               coverage(models)))
        rows.extend(final_rows(k_result, "k", subset, all_final_observed,
                               coverage(models)))

    path = Path(__file__)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (path.parent / "convergence_checkpoints.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checkpoints_rows[0]))
        writer.writeheader()
        writer.writerows(checkpoints_rows)

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), sharex=True)
    x = np.asarray([n for n, _ in plotted])
    for axis, (title, left, right, ratio_key) in zip(axes, COMPARISONS):
        axis.plot(x, [result[left] for _, result in plotted], "o-",
                  markersize=3.5, linewidth=1.4, label=left.replace("_", " "))
        axis.plot(x, [result[right] for _, result in plotted], "s--",
                  markersize=3.2, linewidth=1.4, label=right.replace("_", " "))
        final_ratio = plotted[-1][1]["ratios"][ratio_key]
        axis.set_title(f"{title.replace('_', ' ').title()}\nfinal ratio = {final_ratio:.3f}")
        axis.set_xlabel("Retained iterations per chain")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Variance estimate for H (disasters$^2$)")
    figure.suptitle("Brooks-Giudici comparisons for expected disasters")
    figure.tight_layout(rect=(0, 0.045, 1, 0.94))
    figure.savefig(path.with_suffix(".pdf"))
    print(f"Fixed models: {fixed_models}; all final eligible: {all_final_eligible}")
    print(f"Final H variance ratio V_hat/W_c (fixed): {fixed_h['ratios']['V_hat_over_W_c']:.4f}")
    print(f"Final k variance ratio V_hat/W_c (fixed): {fixed_k['ratios']['V_hat_over_W_c']:.4f}")


if __name__ == "__main__":
    main()
