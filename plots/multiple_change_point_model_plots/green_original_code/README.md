# Comparison with Green's original Cpt implementation

The Python script [`run_green.py`](run_green.py) runs Peter J. Green's original **Cpt** Fortran program and plots the posterior distribution of the number of change points. Cpt implements the multiple change-point model described in:

Green, P. J. (1995). *Reversible jump Markov chain Monte Carlo computation and Bayesian model determination*. Biometrika, 82(4), 711–732.

The Cpt source is **not included in this repository**. To reproduce the comparison:

1. Download the Cpt archive from [Green's official website](https://people.maths.bris.ac.uk/~mapjg/Cpt/).
2. Extract its contents into a directory named `Cpt` beside `run_green.py`. In particular, the paths `green_original_code/Cpt/cpt.f` and `green_original_code/Cpt/coal.dat` must exist; do not leave the files inside an extra nested directory.
3. Install `gcc`, `gfortran`, Python, NumPy, and Matplotlib. The script compiles Cpt in a temporary directory, so no separate manual compilation is needed.
4. From the repository root, run:

   ```bash
   python3 plots/multiple_change_point_model_plots/green_original_code/run_green.py
   ```

The script writes `posterior_k.pdf`, `posterior_k.csv`, `k_samples.csv`, `cpt_run.log`, and `run_info.json` in this directory. It uses 4,000 burn-in sweeps and retains 40,000 subsequent samples.

The Cpt source is Peter J. Green's third-party work. The Python comparison script and generated outputs are part of this repository; the original Cpt distribution must be obtained separately from the author.
