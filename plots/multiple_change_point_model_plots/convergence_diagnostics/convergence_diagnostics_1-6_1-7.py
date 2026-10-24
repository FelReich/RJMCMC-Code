"""Print model 8's contribution and the 1-6 and 1-7 diagnostics."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BaseRJMCMC, baseline_chains  # noqa: E402


def main():
    histories = []
    for draws in baseline_chains().values():
        states = [draw.state for draw in draws]
        # two_way_anova expects an initial state and then discards it.
        histories.append([states[0]] + states)

    result_1_8 = BaseRJMCMC.two_way_anova(
        histories, lambda state: state.H, models=range(1, 9)
    )
    means = result_1_8["means"]
    model_8_contribution = sum(
        (cell_means[8] - chain_mean) ** 2
        for cell_means, chain_mean in zip(means["chain_model"], means["chain"])
    ) / (result_1_8["n_chains"] * (len(result_1_8["selected_models"]) - 1))
    print(f"Model 8 contribution to B_mW_c (models 1-8): {model_8_contribution:.3f}\n")

    print(f"{'Models':<8} {'B_m':>10} {'B_mW_c':>10} {'B_m/B_mW_c':>12}")
    for upper in (6, 7):
        result = BaseRJMCMC.two_way_anova(
            histories, lambda state: state.H, models=range(1, upper + 1)
        )
        ratio = result["ratios"]["B_m_over_B_m_W_c"]
        print(
            f"1-{upper:<6} {result['B_m']:>10.4f} "
            f"{result['B_m_W_c']:>10.4f} {ratio:>12.4f}"
        )


if __name__ == "__main__":
    main()
