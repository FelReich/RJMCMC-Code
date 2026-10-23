"""Run Green's original Cpt sampler and plot its posterior distribution of k.

Run from any directory with ``python3 run_green.py``. Requires gcc, gfortran,
NumPy, and Matplotlib. The original files in Cpt/ are never modified.
"""

import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Cpt"
SEED = 1234
BURN_IN = 4000
RETAINED = 40000
ALPHA = 1
BETA = 200
LAMBDA = 3
K_MAX = 30  # Fixed in Green's cpt.f.

# The original uses integer 1 as an argument to MIN with a real-valued ratio.
# Modern gfortran rejects that mixed-type call; 1.0 is mathematically identical.
OLD_EXPRESSION = "c = min(1.0/min(1,p(1)/p(0)),1.0/min(1,p(kmax-1)/p(kmax)))"
NEW_EXPRESSION = "c = min(1.0/min(1.0,p(1)/p(0)),1.0/min(1.0,p(kmax-1)/p(kmax)))"


def run_command(arguments, directory):
    result = subprocess.run(
        arguments, cwd=directory, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(arguments)}\n"
            f"{result.stdout}"
        )
    return result.stdout


def compile_cpt(directory):
    for name in ("cpt.f", "algama.f", "rgamma.f", "dummy_ieee_chk.f",
                 "sd.c", "coal.dat"):
        shutil.copy2(SOURCE / name, directory / name)
    source = directory / "cpt.f"
    contents = source.read_text(encoding="latin-1")
    if contents.count(OLD_EXPRESSION) != 1:
        raise RuntimeError("The expected compiler compatibility line is missing")
    source.write_text(contents.replace(OLD_EXPRESSION, NEW_EXPRESSION),
                      encoding="latin-1")

    # sd.c calls time() without its header; include it at compile time.
    run_command(["gcc", "-include", "time.h", "-c", "sd.c", "-DSUNF", "-DRETS"], directory)
    run_command([
        "gfortran", "-std=legacy", "-ffixed-line-length-none", "-c",
        "cpt.f", "algama.f", "rgamma.f", "dummy_ieee_chk.f",
    ], directory)
    run_command([
        "gfortran", "-o", "cpt.exe", "cpt.o", "algama.o", "rgamma.o",
        "dummy_ieee_chk.o", "sd.o",
    ], directory)


def save_results(k_samples, log_text):
    indices, counts = np.unique(k_samples, return_counts=True)
    probabilities = counts / len(k_samples)
    mean_k = float(k_samples.mean())
    mode_k = int(indices[np.argmax(counts)])

    with (HERE / "k_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("retained_iteration", "k"))
        writer.writerows(enumerate(k_samples.tolist(), start=1))

    with (HERE / "posterior_k.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("k", "count", "posterior_probability",
                         "posterior_mean_k", "posterior_mode_k"))
        for k, count, probability in zip(indices, counts, probabilities):
            writer.writerow((int(k), int(count), repr(float(probability)),
                             repr(mean_k), mode_k))

    (HERE / "cpt_run.log").write_text(log_text, encoding="utf-8")
    coal_file = SOURCE / "coal.dat"
    metadata = {
        "source_url": "https://people.maths.bris.ac.uk/~mapjg/Cpt/",
        "source_cpt_sha256": hashlib.sha256((SOURCE / "cpt.f").read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(coal_file.read_bytes()).hexdigest(),
        "coal_data_event_count": int(coal_file.read_text().split()[0]),
        "compiler_compatibility_fixes": [
            "Replace two integer 1 arguments to MIN by real 1.0 in a temporary cpt.f copy",
            "Force-include time.h when compiling sd.c",
        ],
        "seed": SEED,
        "burn_in_sweeps": BURN_IN,
        "retained_sweeps": RETAINED,
        "total_sweeps": BURN_IN + RETAINED,
        "sampling_interval": 1,
        "alpha": ALPHA, "beta": BETA, "lambda": LAMBDA,
        "k_max": K_MAX,
        "posterior_mean_k": mean_k,
        "posterior_mode_k": mode_k,
    }
    (HERE / "run_info.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 10,
        "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9, "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })
    figure, axis = plt.subplots(figsize=(7.3, 4.3))
    axis.bar(indices, probabilities, width=0.82, color="#41658a")
    axis.set(xlabel="Number of change points k", ylabel="Posterior probability",
             title="Posterior distribution of k - Green")
    axis.set_xticks(indices)
    axis.set_ylim(0, max(probabilities) * 1.15)
    axis.text(0.98, 0.96, f"Mean = {mean_k:.2f}; mode = {mode_k}",
              transform=axis.transAxes, ha="right", va="top")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(HERE / "posterior_k_green.pdf")
    plt.close(figure)
    return mean_k, mode_k


def main():
    with tempfile.TemporaryDirectory(prefix="green-cpt-") as temporary:
        directory = Path(temporary)
        compile_cpt(directory)
        (directory / "coal").mkdir()
        output = run_command([
            "./cpt.exe", f"-n{RETAINED}", f"-nb{BURN_IN}", "-ns1",
            f"-a{ALPHA}", f"-b{BETA}", f"-l{LAMBDA}",
            f"-seed{SEED}", "coal",
        ], directory)
        sample_files = list((directory / "coal").glob("*.k"))
        if len(sample_files) != 1:
            raise RuntimeError(f"Expected one Cpt .k file, found {sample_files}")
        k_samples = np.loadtxt(sample_files[0], dtype=int, ndmin=1)
        if (k_samples.shape != (RETAINED,) or np.any(k_samples < 0)
                or np.any(k_samples > K_MAX)):
            raise RuntimeError("Cpt did not return 40,000 valid post-burn-in k samples")
        log_file = sample_files[0].with_suffix(".log")
        log_text = log_file.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"mean k\s+([0-9.]+)", log_text)
        if match is None or abs(float(match.group(1)) - k_samples.mean()) > 1e-4:
            raise RuntimeError("Cpt log and sampled k values disagree")

    mean_k, mode_k = save_results(k_samples, log_text)
    print(output.strip())
    print(f"Saved {len(k_samples):,} samples; mean k={mean_k:.4f}, mode k={mode_k}")
    print(f"Figure: {HERE / 'posterior_k_green.pdf'}")


if __name__ == "__main__":
    main()
