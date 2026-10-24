"""PDF table of lag-1 autocorrelation and ESS for the saved baseline H traces.

Use only the five chains with 4,000 burn-in updates and 40,000 retained draws.
Estimate ESS with Geyer's initial monotone sequence of ACF pairs.
"""

import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import RUN_DIR, baseline_chains, style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def autocorrelation_and_ess(values):
    """Compute FFT-based ACF and single-chain initial-monotone-sequence ESS."""
    centered = np.asarray(values, dtype=float) - np.mean(values)
    n = len(centered)
    if n < 2 or not np.all(np.isfinite(centered)) or not np.any(centered):
        raise ValueError("ESS requires at least two finite, nonconstant draws")

    # Julius O. Smith III, Mathematics of the Discrete Fourier Transform
    # (DFT), with Audio Applications, 2nd ed. (2007), "Correlation Analysis":
    # https://www.dsprelated.com/freebooks/mdft/Correlation_Analysis.html
    # Inverse FFT of |FFT|^2 gives autocorrelation. Zero-padding to at least
    # 2*n-1 entries prevents circular wrap-around, giving the usual linear ACF.
    fft_length = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_length)
    products = np.fft.irfft(spectrum * spectrum.conj(), n=fft_length)[:n]
    acf = products / products[0]

    # Geyer, "Practical Markov Chain Monte Carlo," Statistical Science 7(4),
    # 473-483 (1992), Section 3.3. Pair adjacent lags starting at 0, discard
    # pairs from the first nonpositive one onward, and make retained pair sums
    # nonincreasing. This is the initial monotone sequence (IMS) estimator.
    # https://www2.stat.duke.edu/~sschmid/Courses/Stat376/Papers/GeyerStatSci1992.pdf
    pairs = acf[0:2 * (n // 2):2] + acf[1:2 * (n // 2):2]
    nonpositive = np.flatnonzero(pairs <= 0)
    stop = int(nonpositive[0]) if len(nonpositive) else len(pairs)
    monotone_pairs = np.minimum.accumulate(pairs[:stop])
    autocorrelation_time = -1.0 + 2.0 * float(np.sum(monotone_pairs))
    return float(acf[1]), float(n / autocorrelation_time)


def main():
    metadata = json.loads((RUN_DIR / "example_run_metadata.json").read_text())
    burn_in = metadata["burn_in_updates"]
    retained = metadata["retained_states_per_chain"]
    if (burn_in, retained, metadata["total_updates_per_chain"]) != (4_000, 40_000, 44_000):
        raise ValueError("Expected the saved 4,000/40,000 baseline experiment")
    chains = baseline_chains()

    rows = []
    total_ess = 0.0
    for chain_id, draws in chains.items():
        if (len(draws) != retained or draws[0].iteration != burn_in + 1
                or draws[-1].iteration != burn_in + retained):
            raise ValueError(f"Incorrect saved iteration range in {chain_id}")
        values = np.fromiter((draw.H for draw in draws), dtype=float,
                             count=retained)
        lag1, ess = autocorrelation_and_ess(values)
        rows.append((chain_id, f"{lag1:.4f}", f"{ess:.1f}"))
        total_ess += ess
        print(f"{chain_id}: lag-1 = {lag1:.6f}, ESS = {ess:.1f}", flush=True)
    rows.append(("All five", "--", f"{total_ess:.1f}"))

    style()
    figure, axis = plt.subplots(figsize=(6.8, 2.15))
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=("Chain", "Lag-1 autocorrelation", "ESS"),
        colWidths=(0.25, 0.48, 0.27),
        cellLoc="center", colLoc="center", bbox=(0.01, 0.02, 0.98, 0.96),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d5dde5")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e7edf3")
            cell.set_text_props(weight="bold")
        elif row == len(rows):
            cell.set_facecolor("#f1f5f8")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    output = HERE / "autocorrelation_and_ess.pdf"
    figure.savefig(output, pad_inches=0.02)
    plt.close(figure)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
