"""Trace of expected total disasters for the first saved baseline chain."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (  # noqa: E402
    expected_disasters, first_baseline_chain, saved_trace_plot, style,
)


def main():
    style()
    draws = first_baseline_chain()
    figure = saved_trace_plot(draws, statistic=expected_disasters,
                              label="Expected number of disasters")
    axis = figure.axes[0]
    axis.set_title("Expected disasters - baseline chain 1")
    figure.savefig(Path(__file__).with_suffix(".pdf"))


if __name__ == "__main__":
    main()
