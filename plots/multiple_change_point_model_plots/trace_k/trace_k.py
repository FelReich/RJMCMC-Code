"""Trace of k for the first saved baseline chain only."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import first_baseline_chain, saved_trace_plot, style  # noqa: E402


def main():
    style()
    draws = first_baseline_chain()
    figure = saved_trace_plot(draws)
    axis = figure.axes[0]
    axis.set_title("Number of change points - baseline chain 1")
    figure.savefig(Path(__file__).with_suffix(".pdf"))


if __name__ == "__main__":
    main()
