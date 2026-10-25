"""Plot the ACF of H for baseline chain 1 and print its ESS.

Use the saved chain with 4,000 burn-in updates and 40,000 retained draws.
Estimate ESS with Geyer's initial monotone sequence of ACF pairs.
"""

import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common import RUN_DIR, first_baseline_chain, style  # noqa: E402
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
    return acf, float(n / autocorrelation_time)


def main():
    metadata = json.loads((RUN_DIR / "example_run_metadata.json").read_text())
    burn_in = metadata["burn_in_updates"]
    retained = metadata["retained_states_per_chain"]
    if (burn_in, retained, metadata["total_updates_per_chain"]) != (4_000, 40_000, 44_000):
        raise ValueError("Expected the saved 4,000/40,000 baseline experiment")
    draws = first_baseline_chain()
    if (len(draws) != retained or draws[0].iteration != burn_in + 1
            or draws[-1].iteration != burn_in + retained):
        raise ValueError("Incorrect saved iteration range in base_1")
    values = np.fromiter((draw.H for draw in draws), dtype=float,
                         count=retained)
    acf, ess = autocorrelation_and_ess(values)

    style()
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    lags = np.arange(501)
    axis.plot(lags, acf[:501], color="#176b98", linewidth=1.5)
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set(xlabel="Lag", ylabel="Autocorrelation", xlim=(0, 500))
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output = HERE / "autocorrelation_and_ess.pdf"
    figure.savefig(output)
    plt.close(figure)
    print(f"ESS for H in base_1: {ess:.1f}")


if __name__ == "__main__":
    main()
