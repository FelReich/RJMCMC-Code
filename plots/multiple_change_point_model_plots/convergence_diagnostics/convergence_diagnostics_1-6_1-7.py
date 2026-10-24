"""Print the final between-model diagnostic for models 1-6 and 1-7."""

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
