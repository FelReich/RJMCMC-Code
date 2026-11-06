"""Preliminary three-case GCLM comparison; no sampler or move changes.

Run from the Code repository root:
    python3 examples/gclm/dummy/gclm_comparison_test.py

Each of the 15 chains uses 100,000 updates and discards the first 50,000.
Stored draws contain only the numerical quantities needed for analysis.
"""

from collections import Counter
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.special import gammaln, logsumexp
from scipy.stats import norm, rankdata

from examples.gclm.gclm import GCLM, GCLMCalculations, GCLMPriorConfig, GCLMProposalConfig
from examples.gclm.gclm_moves import (
    MeanMove, ContrastMove, MarginMove, CoefficientMove,
    ReversalMove, BirthMove, DeathMove,
)
from examples.gclm.state import Graph, State


N_SAMPLES = 100
N_ITERATIONS = 1_000_000
BURN_IN = 500_000
CHAIN_SEEDS = (101, 202, 303, 404, 505)
DATA_SEEDS = {"simple": 24680, "nonsimple": 13579}
OUTPUT = Path(__file__).resolve().parent / "dummy_outputs" / "comparison_study"
EDGES = tuple((i, j) for i in range(3) for j in range(3) if i != j)
EDGE_INDEX = {edge: index for index, edge in enumerate(EDGES)}
M_SIMPLE = np.array([
    [-1.5, 0.0, 0.0],
    [0.5, -1.5, 0.0],
    [0.6, 0.0, -1.0],
])
M_NONSIMPLE = np.array([
    [-1.5, 0.5, 0.0],
    [0.5, -1.5, 0.0],
    [0.6, -0.7, -1.0],
])
G_SIMPLE = frozenset({(0, 1), (0, 2)})
G_NONSIMPLE = frozenset({(0, 1), (1, 0), (0, 2), (1, 2)})
SIMPLE_STARTS = (
    frozenset(),
    frozenset({(0, 1)}),
    G_SIMPLE,
    frozenset({(1, 0), (2, 1)}),
    frozenset({(0, 1), (0, 2), (1, 2)}),
)
GENERAL_STARTS = (
    frozenset(),
    frozenset({(0, 1)}),
    G_SIMPLE,
    G_NONSIMPLE,
    frozenset(EDGES),
)
CASES = (
    ("A_simple_on_simple", "simple", "simple", SIMPLE_STARTS),
    ("B_general_on_simple", "simple", "general", SIMPLE_STARTS),
    ("C_general_on_nonsimple", "nonsimple", "general", GENERAL_STARTS),
)
PRIOR_SETTINGS = dict(
    lambda_p=3.0, s_r=1.0, mu_eta=0.0, s_eta=0.75,
    tau=1.0, m_0=0.0, s_mu=1.0,
)
PROPOSAL_SETTINGS = dict(
    sigma_r=0.15, sigma_eta=0.12, sigma_b=0.30, gamma=0.5,
    move_weights={name: 1.0 for name in GCLM.MOVE_NAMES},
)
MOVE_CLASSES = dict(zip(
    GCLM.MOVE_NAMES,
    (MeanMove, ContrastMove, MarginMove, CoefficientMove,
     ReversalMove, BirthMove, DeathMove),
))


def save_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_pdf(figure, path):
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def edge_mask(edges):
    return sum(1 << EDGE_INDEX[edge] for edge in edges)


def mask_edges(mask):
    return tuple(edge for index, edge in enumerate(EDGES) if int(mask) & (1 << index))


def graph_label(mask):
    return ", ".join(f"{i+1}→{j+1}" for i, j in mask_edges(mask)) or "∅"


def graph_prior(space):
    """Truncated Poisson prior on k; conditional graphs are uniform."""
    upper = 3 if space == "simple" else 6
    k = np.arange(upper + 1)
    log_mass = k * np.log(PRIOR_SETTINGS["lambda_p"]) - gammaln(k + 1)
    return k, np.exp(log_mass - logsumexp(log_mass))


def generating_setup(model_name):
    M = M_SIMPLE if model_name == "simple" else M_NONSIMPLE
    edges = G_SIMPLE if model_name == "simple" else G_NONSIMPLE
    nonzero_edges = frozenset((source, target) for source, target in EDGES
                              if M[target, source] != 0)
    if nonzero_edges != edges:
        raise RuntimeError("Generating graph does not match the drift's off-diagonal entries")
    spectral_abscissa = float(np.max(np.linalg.eigvals(M).real))
    if not spectral_abscissa < 0:
        raise RuntimeError("Generating drift is not stable")
    sigma = solve_continuous_lyapunov(M, -2.0 * np.eye(3))
    sigma = (sigma + sigma.T) / 2
    smallest_eigenvalue = float(np.linalg.eigvalsh(sigma).min())
    if smallest_eigenvalue <= 0:
        raise RuntimeError("Generating covariance is not positive definite")
    relative_residual = float(
        np.linalg.norm(M @ sigma + sigma @ M.T + 2 * np.eye(3), ord="fro")
        / (np.linalg.norm(M @ sigma, ord="fro")
           + np.linalg.norm(sigma @ M.T, ord="fro")
           + np.linalg.norm(2 * np.eye(3), ord="fro"))
    )
    if relative_residual > 1e-10:
        raise RuntimeError("Generating Lyapunov residual is too large")
    observations = np.random.default_rng(DATA_SEEDS[model_name]).multivariate_normal(
        np.zeros(3), sigma, size=N_SAMPLES
    )
    return dict(M=M, sigma=sigma, edges=edges, data=observations,
                spectral_abscissa=spectral_abscissa,
                smallest_covariance_eigenvalue=smallest_eigenvalue,
                relative_lyapunov_residual=relative_residual)


def initial_state(edges, seed):
    rng = np.random.default_rng(seed + 10_000)
    graph = Graph(3, edges)
    return State(
        graph=graph,
        mu=rng.normal(0, 0.2, size=3),
        r=rng.normal(0, 0.35, size=2),
        eta=float(np.exp(rng.normal(0, 0.35))),
        b=rng.normal(0, 0.5, size=graph.k),
    )


class CountingCalculations(GCLMCalculations):
    """Count failed candidate calculations without changing their behavior."""

    def __init__(self, observations):
        super().__init__(observations)
        self.calculation_count = 0
        self.failed_calculations = 0

    def _calculate_drift(self, state, key):
        self.calculation_count += 1
        try:
            return super()._calculate_drift(state, key)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            self.failed_calculations += 1
            raise


def run_chain(case_name, chain_id, seed, start_edges, graph_space, data):
    """Record numerical draws directly from step(), without a State history."""
    initial = initial_state(start_edges, seed)
    initial.validate_for_model(3, graph_space)
    prior = GCLMPriorConfig(graph_space=graph_space, **PRIOR_SETTINGS)
    proposal = GCLMProposalConfig(**PROPOSAL_SETTINGS)
    calculations = CountingCalculations(data)
    moves = {name: cls(calculations, prior, proposal)
             for name, cls in MOVE_CLASSES.items()}
    sampler = GCLM(moves, calculations, prior, proposal,
                   initial_state=initial, seed=seed)

    retained = N_ITERATIONS - BURN_IN
    draws = dict(
        k=np.empty(retained, dtype=np.uint8),
        graph_mask=np.empty(retained, dtype=np.uint8),
        M=np.empty((retained, 3, 3)),
        Sigma=np.empty((retained, 3, 3)),
        K=np.empty((retained, 3, 3)),
        eta=np.empty(retained),
        log_likelihood=np.empty(retained),
    )
    proposed = Counter()
    accepted = Counter()
    retained_proposed = Counter()
    retained_accepted = Counter()
    graph_changes = 0
    maximum_residual = 0.0
    maximum_skew_error = 0.0
    previous_graph = initial.graph
    start = perf_counter()
    for iteration in range(N_ITERATIONS):
        state, move, did_accept = sampler.step()
        proposed[move] += 1
        accepted[move] += int(did_accept)
        if iteration >= BURN_IN:
            index = iteration - BURN_IN
            retained_proposed[move] += 1
            retained_accepted[move] += int(did_accept)
            graph_changes += int(state.graph != previous_graph)
            calculation = calculations.drift(state)
            draws["k"][index] = state.graph.k
            draws["graph_mask"][index] = edge_mask(state.graph.edges)
            draws["M"][index] = calculation.M
            draws["Sigma"][index] = calculation.covariance
            K = calculation.M @ calculation.covariance + np.eye(3)
            draws["K"][index] = K
            draws["eta"][index] = state.eta
            draws["log_likelihood"][index] = calculations.log_likelihood(state)
            maximum_residual = max(maximum_residual,
                                   calculation.relative_lyapunov_residual)
            maximum_skew_error = max(maximum_skew_error,
                                     float(np.abs(K + K.T).max()))
        previous_graph = state.graph
    seconds = perf_counter() - start
    if maximum_skew_error > 1e-8 or not all(np.all(np.isfinite(v)) for v in draws.values()):
        raise RuntimeError(f"Invalid retained draws or K symmetry in {case_name} chain {chain_id}")

    folder = OUTPUT / case_name
    np.savez_compressed(folder / f"chain_{chain_id}_retained.npz", **draws)
    result = dict(
        chain_id=chain_id, seed=seed, initial_graph=graph_label(edge_mask(start_edges)),
        initial_edges=[list(edge) for edge in sorted(start_edges)],
        runtime_seconds=seconds, distinct_graphs=int(np.unique(draws["graph_mask"]).size),
        graph_changes=graph_changes, graph_changes_per_retained_iteration=graph_changes / retained,
        calculation_count=calculations.calculation_count,
        failed_calculations=calculations.failed_calculations,
        maximum_relative_lyapunov_residual=maximum_residual,
        maximum_skew_symmetry_error=maximum_skew_error,
        acceptance={name: dict(
            proposed=proposed[name], accepted=accepted[name],
            rate=accepted[name] / proposed[name] if proposed[name] else None,
            retained_proposed=retained_proposed[name],
            retained_accepted=retained_accepted[name],
        ) for name in GCLM.MOVE_NAMES},
    )
    print(f"  chain {chain_id}: {seconds:.1f}s, mean k={draws['k'].mean():.3f}, "
          f"graphs={result['distinct_graphs']}, changes={graph_changes}", flush=True)
    return result


def load_chains(case_name):
    """Load only numerical retained draws; no sampler State histories."""
    chains = []
    for chain_id in range(1, len(CHAIN_SEEDS) + 1):
        with np.load(OUTPUT / case_name / f"chain_{chain_id}_retained.npz") as file:
            chains.append({key: file[key] for key in file.files})
    return chains


def split_rank_rhat(chains):
    """Maximum of rank-normalized split and folded split R-hat."""
    values = np.asarray(chains, dtype=float)
    n_half = values.shape[1] // 2
    split = np.concatenate((values[:, :n_half], values[:, -n_half:]), axis=0)

    def transform(array):
        ranks = rankdata(array.ravel(), method="average")
        return norm.ppf((ranks - 0.375) / (len(ranks) + 0.25)).reshape(array.shape)

    def rhat(array):
        n = array.shape[1]
        within = np.var(array, axis=1, ddof=1).mean()
        between = n * np.var(array.mean(axis=1), ddof=1)
        if within == 0:
            return 1.0 if between == 0 else float("inf")
        return float(np.sqrt(((n - 1) / n * within + between / n) / within))

    bulk = rhat(transform(split))
    folded = rhat(transform(np.abs(split - np.median(split))))
    return max(bulk, folded), bulk, folded


def single_chain_ess(values):
    """Approximate FFT ESS using the initial positive autocorrelation pairs."""
    centered = np.asarray(values, dtype=float) - np.mean(values)
    denominator = float(centered @ centered)
    if denominator == 0:
        return None
    n = len(centered)
    padding = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(centered, n=padding)
    autocovariance = np.fft.irfft(transformed * transformed.conj(), n=padding)
    acf = autocovariance[:n] / denominator
    pairs = acf[1:-1:2] + acf[2::2]
    stop = np.flatnonzero(pairs <= 0)
    positive = pairs[:stop[0]] if stop.size else pairs
    return float(min(n, n / (1 + 2 * positive.sum())))


def diagnostic_series(chains):
    """Scalar functions of retained draws, including every drift entry."""
    series = {
        "k": np.stack([chain["k"] for chain in chains]).astype(float),
        "log_likelihood": np.stack([chain["log_likelihood"] for chain in chains]),
        "eta": np.stack([chain["eta"] for chain in chains]),
    }
    for index, (source, target) in enumerate(EDGES):
        series[f"edge_{source+1}_to_{target+1}"] = np.stack([
            ((chain["graph_mask"] >> index) & 1).astype(float) for chain in chains
        ])
    for i in range(3):
        for j in range(3):
            series[f"M_{i+1}_{j+1}"] = np.stack([chain["M"][:, i, j] for chain in chains])
        for j in range(i, 3):
            series[f"Sigma_{i+1}_{j+1}"] = np.stack([
                chain["Sigma"][:, i, j] for chain in chains
            ])
            if i != j:
                series[f"K_{i+1}_{j+1}"] = np.stack([
                    chain["K"][:, i, j] for chain in chains
                ])
    return series


def diagnostics_table(chains, total_seconds):
    results = []
    for name, values in diagnostic_series(chains).items():
        rhat, rank_split, folded_split = split_rank_rhat(values)
        chain_ess = [single_chain_ess(chain) for chain in values]
        finite_ess = [value for value in chain_ess if value is not None]
        minimum_ess = min(finite_ess) if finite_ess else None
        total_ess = sum(finite_ess) if finite_ess else None
        results.append(dict(
            quantity=name, rhat=rhat, rank_split_rhat=rank_split,
            folded_rank_split_rhat=folded_split,
            minimum_single_chain_ess=minimum_ess,
            summed_approximate_ess=total_ess,
            approximate_ess_per_second=total_ess / total_seconds if total_ess else None,
            single_chain_ess=chain_ess,
            chain_means=values.mean(axis=1).tolist(),
            flag=bool(rhat > 1.05 or minimum_ess is None or minimum_ess < 100),
        ))
    return results


def plot_graph_size(folder, prior_k, prior_mass, posterior, true_k):
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    values = np.arange(len(prior_mass))
    axis.bar(values - 0.19, prior_mass, width=0.38, color="0.65", label="Prior")
    axis.bar(values + 0.19, posterior, width=0.38, color="tab:blue", label="Posterior")
    axis.axvline(true_k, color="tab:red", linestyle="--", linewidth=1.2,
                 label=f"Generating k = {true_k}")
    axis.set(xlabel="Number of off-diagonal edges k", ylabel="Probability",
             xticks=values, ylim=(0, max(np.max(prior_mass), np.max(posterior)) * 1.16))
    axis.legend()
    save_pdf(figure, folder / "prior_vs_posterior_k.pdf")


def edge_matrix(probabilities):
    matrix = np.full((3, 3), np.nan)
    for probability, (source, target) in zip(probabilities, EDGES):
        matrix[target, source] = probability
    return matrix


def draw_matrix(axis, matrix, title, vmin, vmax, cmap="RdBu_r", labels=True):
    image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set(title=title, xticks=range(3), yticks=range(3))
    axis.set_xticklabels(("1", "2", "3"))
    axis.set_yticklabels(("1", "2", "3"))
    if labels:
        for i in range(3):
            for j in range(3):
                if np.isfinite(matrix[i, j]):
                    axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                              fontsize=8, color="black")
    return image


def plot_edge_heatmap(folder, probabilities):
    figure, axis = plt.subplots(figsize=(4.5, 4.1))
    image = draw_matrix(axis, edge_matrix(probabilities),
                        "Posterior edge-inclusion probabilities", 0, 1,
                        cmap="viridis")
    axis.set(xlabel="Source vertex", ylabel="Target vertex")
    figure.colorbar(image, ax=axis, shrink=0.78)
    save_pdf(figure, folder / "edge_inclusion_heatmap.pdf")


def plot_network(folder, probabilities):
    positions = {0: np.array([0.0, 0.9]),
                 1: np.array([-0.9, -0.55]), 2: np.array([0.9, -0.55])}
    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    axis.set_xlim(-1.45, 1.45)
    axis.set_ylim(-1.1, 1.35)
    axis.set_aspect("equal")
    axis.axis("off")
    for source, position in positions.items():
        axis.add_patch(Circle(position, 0.15, facecolor="white", edgecolor="black", lw=1.5))
        axis.text(*position, str(source + 1), ha="center", va="center", zorder=5)
    for probability, (source, target) in zip(probabilities, EDGES):
        if probability <= 0.5:
            continue
        reverse_probability = probabilities[EDGE_INDEX[(target, source)]]
        curvature = 0.15 if reverse_probability > 0.5 else 0.0
        start, end = positions[source], positions[target]
        axis.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14,
            connectionstyle=f"arc3,rad={curvature}", shrinkA=14, shrinkB=14,
            linewidth=1 + 2 * probability, color="tab:blue",
        ))
        delta = end - start
        perpendicular = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        label_position = (start + end) / 2 + (0.19 if curvature else 0.11) * perpendicular
        axis.text(*label_position, f"{probability:.2f}", fontsize=8,
                  ha="center", va="center", bbox=dict(facecolor="white", alpha=0.8,
                                                        edgecolor="none", pad=1))
    axis.set_title("Edges with posterior inclusion probability > 0.5")
    save_pdf(figure, folder / "edge_network_gt_0_5.pdf")


def plot_k_traces(folder, chains):
    figure, axes = plt.subplots(5, 1, figsize=(8.5, 7.2), sharex=True, sharey=True)
    for index, (axis, chain) in enumerate(zip(axes, chains), start=1):
        x = np.arange(0, len(chain["k"]), 10)
        axis.plot(x + 1, chain["k"][::10], lw=0.55)
        axis.set_ylabel(f"Chain {index}")
        axis.set_ylim(-0.3, 6.3)
    axes[-1].set_xlabel("Retained iteration (after burn-in)")
    save_pdf(figure, folder / "k_traces.pdf")


def cumulative_curves(axis, chains, extractor, ylabel, truth=None):
    n = len(chains[0]["k"])
    x = np.arange(1, n + 1)
    pooled = np.zeros(n)
    for index, chain in enumerate(chains, start=1):
        values = np.asarray(extractor(chain), dtype=float)
        cumulative = np.cumsum(values)
        pooled += cumulative
        axis.plot(x[999:], (cumulative / x)[999:], lw=0.9, alpha=0.7,
                  label=f"Chain {index}")
    axis.plot(x[999:], (pooled / (len(chains) * x))[999:], color="black",
              lw=1.8, label="Pooled")
    if truth is not None:
        axis.axhline(truth, color="tab:red", linestyle="--", lw=1.1,
                     label="Generating value")
    axis.set(xlim=(1000, n), xlabel="Retained iterations per chain", ylabel=ylabel)
    axis.grid(alpha=0.2)


def plot_cumulative(folder, chains, true_mask, true_k, K_true):
    for filename, extractor, ylabel, truth in (
        ("cumulative_mean_k.pdf", lambda c: c["k"], "Cumulative E[k | y]", true_k),
        ("cumulative_true_graph.pdf",
         lambda c: c["graph_mask"] == true_mask,
         "Cumulative P(generating graph | y)", None),
    ):
        figure, axis = plt.subplots(figsize=(8.5, 4.4))
        cumulative_curves(axis, chains, extractor, ylabel, truth)
        axis.legend(ncol=2, fontsize=8)
        save_pdf(figure, folder / filename)
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True)
    for axis, (i, j) in zip(axes, ((0, 1), (0, 2), (1, 2))):
        cumulative_curves(axis, chains, lambda c, i=i, j=j: c["K"][:, i, j],
                          f"Cumulative mean K[{i},{j}]", K_true[i, j])
    axes[0].legend(ncol=3, fontsize=7)
    save_pdf(figure, folder / "skew_cumulative_means.pdf")


def plot_covariance(folder, sigma_true, empirical, posterior):
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.6))
    for axis, matrix, title in zip(
        axes, (sigma_true, empirical, posterior),
        ("Generating covariance", "Empirical covariance", "Posterior mean covariance"),
    ):
        # A signed scale is needed: the non-simple generating covariance has
        # negative off-diagonal entries. Keep it fixed across experiments.
        image = draw_matrix(axis, matrix, title, -1.5, 1.5)
    figure.colorbar(image, ax=axes, shrink=0.72)
    figure.savefig(folder / "covariance_heatmaps.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_drift(folder, M_true, posterior):
    figure, axes = plt.subplots(1, 2, figsize=(7.7, 3.5))
    for axis, matrix, title in zip(
        axes, (M_true, posterior), ("Generating drift", "Posterior mean drift"),
    ):
        image = draw_matrix(axis, matrix, title, -2, 2)
    figure.colorbar(image, ax=axes, shrink=0.72)
    figure.savefig(folder / "drift_heatmaps.pdf", bbox_inches="tight")
    plt.close(figure)


def selected_drift_entries(true_edges):
    entries = [(target, source) for source, target in sorted(true_edges)]
    if true_edges == G_SIMPLE:
        entries.append((0, 1))  # Reverse of the 1→2 generating edge.
    return entries


def plot_scalar_histograms(folder, values_by_entry, truths, filename, zero_atom=False):
    entries = list(values_by_entry)
    figure, axes = plt.subplots(1, len(entries), figsize=(3.25 * len(entries), 3.4))
    axes = np.atleast_1d(axes)
    for axis, entry in zip(axes, entries):
        values = np.asarray(values_by_entry[entry])
        nonzero = values[values != 0]
        if len(nonzero):
            axis.hist(nonzero, bins=35, weights=np.full(len(nonzero), 1 / len(values)),
                      color="tab:blue", alpha=0.8)
        if zero_atom:
            mass = 1 - len(nonzero) / len(values)
            if mass:
                width = max(np.ptp(values) * 0.015, 0.02)
                axis.bar(0, mass, width=width, color="tab:orange", alpha=0.9,
                         label=f"P(exactly 0)={mass:.2f}")
                axis.legend(fontsize=7)
        if entry in truths:
            axis.axvline(truths[entry], color="tab:red", linestyle="--", lw=1.1)
        axis.set(title=f"{filename.split('_')[0]}[{entry[0]},{entry[1]}]",
                 xlabel="Value", ylabel="Posterior probability per bin")
    save_pdf(figure, folder / filename)


def plot_drift_traces(folder, chains, entries):
    figure, axes = plt.subplots(len(entries), 1, figsize=(8.5, 2.35 * len(entries)),
                                sharex=True)
    axes = np.atleast_1d(axes)
    for axis, (i, j) in zip(axes, entries):
        for chain_index, chain in enumerate(chains, start=1):
            x = np.arange(0, len(chain["k"]), 25)
            axis.plot(x + 1, chain["M"][::25, i, j], lw=0.55,
                      alpha=0.65, label=f"Chain {chain_index}")
        axis.set_ylabel(f"M[{i},{j}]")
        axis.grid(alpha=0.15)
    axes[0].legend(ncol=5, fontsize=7)
    axes[-1].set_xlabel("Retained iteration (after burn-in)")
    save_pdf(figure, folder / "drift_selected_traces.pdf")


def analyze_case(case_name, model_name, graph_space, generation, chain_stats):
    folder = OUTPUT / case_name
    chains = load_chains(case_name)
    total_seconds = sum(item["runtime_seconds"] for item in chain_stats)
    true_mask = edge_mask(generation["edges"])
    masks = np.concatenate([chain["graph_mask"] for chain in chains])
    k = np.concatenate([chain["k"] for chain in chains])
    M = np.concatenate([chain["M"] for chain in chains])
    Sigma = np.concatenate([chain["Sigma"] for chain in chains])
    K = np.concatenate([chain["K"] for chain in chains])
    prior_k, prior_mass = graph_prior(graph_space)
    posterior_k = np.bincount(k, minlength=len(prior_k))[:len(prior_k)] / len(k)
    graph_counts = Counter(int(mask) for mask in masks)
    true_graph_probability = graph_counts[true_mask] / len(masks)
    inclusion = np.array([np.mean((masks >> index) & 1)
                          for index in range(len(EDGES))])
    sample_covariance = np.cov(generation["data"], rowvar=False, bias=True)
    M_mean = M.mean(axis=0)
    Sigma_mean = Sigma.mean(axis=0)
    K_mean = K.mean(axis=0)
    M_intervals = np.quantile(M, [0.025, 0.975], axis=0)
    Sigma_intervals = np.quantile(Sigma, [0.025, 0.975], axis=0)
    K_intervals = np.quantile(K, [0.025, 0.975], axis=0)
    between_chain_sd_M = np.std(
        np.stack([chain["M"].mean(axis=0) for chain in chains]), axis=0, ddof=1
    )
    diagnostics = diagnostics_table(chains, total_seconds)
    diagnostic_map = {item["quantity"]: item for item in diagnostics}
    top_graphs = graph_counts.most_common(10)
    K_true = generation["M"] @ generation["sigma"] + np.eye(3)

    save_csv(folder / "prior_posterior_k.csv",
             ("k", "prior_probability", "posterior_probability"),
             zip(prior_k, prior_mass, posterior_k))
    save_csv(folder / "edge_inclusion.csv",
             ("source_vertex", "target_vertex", "true_edge", "posterior_probability"),
             ((i + 1, j + 1, int((i, j) in generation["edges"]), probability)
              for probability, (i, j) in zip(inclusion, EDGES)))
    save_csv(folder / "top_graphs.csv",
             ("rank", "graph", "edge_count", "draws", "posterior_probability"),
             ((rank, graph_label(mask), int(mask).bit_count(), count, count / len(masks))
              for rank, (mask, count) in enumerate(top_graphs, start=1)))
    save_csv(folder / "chain_summary.csv",
             ("chain", "seed", "initial_graph", "runtime_seconds", "distinct_graphs",
              "graph_changes", "graph_changes_per_retained_iteration",
              "calculation_count", "failed_calculations", "maximum_lyapunov_residual",
              "maximum_skew_symmetry_error"),
             ((item["chain_id"], item["seed"], item["initial_graph"],
               item["runtime_seconds"], item["distinct_graphs"],
               item["graph_changes"], item["graph_changes_per_retained_iteration"],
               item["calculation_count"], item["failed_calculations"],
               item["maximum_relative_lyapunov_residual"],
               item["maximum_skew_symmetry_error"]) for item in chain_stats))
    save_csv(folder / "acceptance_rates.csv",
             ("chain", "move", "proposed_all", "accepted_all", "rate_all",
              "proposed_retained", "accepted_retained", "rate_retained"),
             ((item["chain_id"], name, counts["proposed"], counts["accepted"],
               counts["rate"], counts["retained_proposed"],
               counts["retained_accepted"],
               counts["retained_accepted"] / counts["retained_proposed"]
               if counts["retained_proposed"] else None)
              for item in chain_stats for name, counts in item["acceptance"].items()))
    save_csv(folder / "diagnostics.csv",
             ("quantity", "rhat", "rank_split_rhat", "folded_rank_split_rhat",
              "minimum_single_chain_ess", "summed_approximate_ess",
              "approximate_ess_per_second", "flag"),
             ((item["quantity"], item["rhat"], item["rank_split_rhat"],
               item["folded_rank_split_rhat"], item["minimum_single_chain_ess"],
               item["summed_approximate_ess"], item["approximate_ess_per_second"],
               item["flag"]) for item in diagnostics))
    save_csv(folder / "covariance_entries.csv",
             ("row", "column", "generating", "empirical", "posterior_mean",
              "credible_2_5_percent", "credible_97_5_percent"),
             ((i + 1, j + 1, generation["sigma"][i, j], sample_covariance[i, j],
               Sigma_mean[i, j], Sigma_intervals[0, i, j], Sigma_intervals[1, i, j])
              for i in range(3) for j in range(i, 3)))
    save_csv(folder / "drift_entries.csv",
             ("row", "column", "generating", "posterior_mean",
              "credible_2_5_percent", "credible_97_5_percent",
              "between_chain_sd_of_means", "point_mass_at_zero"),
             ((i + 1, j + 1, generation["M"][i, j], M_mean[i, j],
               M_intervals[0, i, j], M_intervals[1, i, j],
               between_chain_sd_M[i, j], float(np.mean(M[:, i, j] == 0)))
              for i in range(3) for j in range(3)))
    save_csv(folder / "skew_entries.csv",
             ("row", "column", "generating", "posterior_mean",
              "credible_2_5_percent", "credible_97_5_percent"),
             ((i + 1, j + 1, K_true[i, j], K_mean[i, j],
               K_intervals[0, i, j], K_intervals[1, i, j])
              for i, j in ((0, 1), (0, 2), (1, 2))))

    checkpoints = []
    for chain_id, chain in enumerate(chains, start=1):
        for count in (10_000, 25_000, 40_000, 50_000):
            if count <= len(chain["k"]):
                checkpoints.append((chain_id, count,
                                    float(chain["k"][:count].mean()),
                                    float(np.mean(chain["graph_mask"][:count] == true_mask)),
                                    float(chain["M"][:count, 1, 0].mean()),
                                    float(chain["K"][:count, 0, 1].mean())))
    save_csv(folder / "cumulative_checkpoints.csv",
             ("chain", "retained_count", "mean_k", "true_graph_probability",
              "mean_M_2_1", "mean_K_1_2"), checkpoints)

    plot_graph_size(folder, prior_k, prior_mass, posterior_k, len(generation["edges"]))
    plot_edge_heatmap(folder, inclusion)
    plot_network(folder, inclusion)
    plot_k_traces(folder, chains)
    plot_cumulative(folder, chains, true_mask, len(generation["edges"]), K_true)
    plot_covariance(folder, generation["sigma"], sample_covariance, Sigma_mean)
    plot_drift(folder, generation["M"], M_mean)
    selected = selected_drift_entries(generation["edges"])
    plot_scalar_histograms(
        folder, {entry: M[:, entry[0], entry[1]] for entry in selected},
        {entry: generation["M"][entry] for entry in selected},
        "drift_selected_histograms.pdf", zero_atom=True,
    )
    plot_drift_traces(folder, chains, selected)
    plot_scalar_histograms(
        folder, {(i, j): K[:, i, j] for i, j in ((0, 1), (0, 2), (1, 2))},
        {(i, j): K_true[i, j] for i, j in ((0, 1), (0, 2), (1, 2))},
        "skew_histograms.pdf", zero_atom=True,
    )

    summary = dict(
        case=case_name, generating_model=model_name, inference_graph_space=graph_space,
        n_samples=N_SAMPLES, n_iterations=N_ITERATIONS, burn_in=BURN_IN,
        retained_per_chain=N_ITERATIONS - BURN_IN, data_seed=DATA_SEEDS[model_name],
        chain_seeds=list(CHAIN_SEEDS), prior_settings=PRIOR_SETTINGS,
        proposal_settings=PROPOSAL_SETTINGS,
        generating_M=generation["M"].tolist(),
        generating_sigma=generation["sigma"].tolist(),
        generating_K=K_true.tolist(),
        generating_edges=[list(edge) for edge in sorted(generation["edges"])],
        generating_spectral_abscissa=generation["spectral_abscissa"],
        generating_min_covariance_eigenvalue=generation["smallest_covariance_eigenvalue"],
        generating_relative_lyapunov_residual=generation["relative_lyapunov_residual"],
        prior_graph_size_probabilities=prior_mass.tolist(),
        posterior_graph_size_probabilities=posterior_k.tolist(),
        posterior_mean_k=float(k.mean()),
        posterior_k_95_percent_credible_interval=np.quantile(k, [0.025, 0.975]).tolist(),
        true_graph_posterior_probability=true_graph_probability,
        edge_inclusion_probabilities={f"{i+1}->{j+1}": float(value)
                                      for value, (i, j) in zip(inclusion, EDGES)},
        top_ten_graphs=[dict(graph=graph_label(mask), draws=count,
                             probability=count / len(masks)) for mask, count in top_graphs],
        distinct_graphs=len(graph_counts),
        graph_changes_per_retained_iteration=sum(item["graph_changes"] for item in chain_stats)
        / (len(chain_stats) * (N_ITERATIONS - BURN_IN)),
        empirical_covariance=sample_covariance.tolist(),
        posterior_mean_covariance=Sigma_mean.tolist(),
        posterior_covariance_interval=Sigma_intervals.tolist(),
        posterior_covariance_frobenius_error=float(np.linalg.norm(
            Sigma_mean - generation["sigma"], ord="fro")),
        empirical_covariance_frobenius_error=float(np.linalg.norm(
            sample_covariance - generation["sigma"], ord="fro")),
        posterior_mean_drift=M_mean.tolist(),
        posterior_drift_interval=M_intervals.tolist(),
        posterior_drift_frobenius_error=float(np.linalg.norm(
            M_mean - generation["M"], ord="fro")),
        between_chain_sd_of_mean_drift=between_chain_sd_M.tolist(),
        posterior_mean_skew=K_mean.tolist(),
        posterior_skew_interval=K_intervals.tolist(),
        chain_sampling_seconds=total_seconds,
        chains=chain_stats,
        diagnostics=diagnostics,
        flagged_quantities=[item["quantity"] for item in diagnostics if item["flag"]],
        diagnostic_flag_rule="R-hat > 1.05 or minimum single-chain approximate ESS < 100",
        ess_method="FFT autocorrelation, initial positive pairs, per-chain approximate ESS",
    )
    save_json(folder / "summary.json", summary)
    print(f"  posterior mean k={summary['posterior_mean_k']:.3f}; "
          f"P(true graph)={true_graph_probability:.4f}; "
          f"M error={summary['posterior_drift_frobenius_error']:.3f}; "
          f"Sigma error={summary['posterior_covariance_frobenius_error']:.3f}", flush=True)
    print(f"  max R-hat M={max(diagnostic_map[f'M_{i+1}_{j+1}']['rhat'] for i in range(3) for j in range(3)):.3f}; "
          f"max R-hat Sigma={max(diagnostic_map[f'Sigma_{i+1}_{j+1}']['rhat'] for i in range(3) for j in range(i, 3)):.3f}", flush=True)
    return summary


def compare_cases(summaries, simple_generation):
    shared = OUTPUT / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    a, b, c = (summaries[case[0]] for case in CASES)

    # A and B have the same observations and generating drift. Only the
    # admissible graph space (hence its truncated graph prior) changes.
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(7)
    axis.bar(x - 0.2, np.pad(a["posterior_graph_size_probabilities"], (0, 3)),
             width=0.4, label="A: simple inference", color="tab:blue")
    axis.bar(x + 0.2, b["posterior_graph_size_probabilities"],
             width=0.4, label="B: general inference", color="tab:orange")
    axis.axvline(len(G_SIMPLE), color="tab:red", linestyle="--", lw=1.1,
                 label="Generating k")
    axis.set(xlabel="Number of off-diagonal edges k", ylabel="Posterior probability",
             xticks=x)
    axis.legend()
    save_pdf(figure, shared / "A_vs_B_posterior_k.pdf")

    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for axis, summary, title in zip(axes, (a, b), ("A: simple", "B: general")):
        probabilities = [summary["edge_inclusion_probabilities"][f"{i+1}->{j+1}"]
                         for i, j in EDGES]
        image = draw_matrix(axis, edge_matrix(probabilities), title, 0, 1,
                            cmap="viridis")
        axis.set(xlabel="Source vertex", ylabel="Target vertex")
    figure.colorbar(image, ax=axes, shrink=0.75)
    figure.savefig(shared / "A_vs_B_edge_inclusion.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(13.1, 3.5))
    for axis, matrix, title in zip(
        axes,
        (simple_generation["sigma"], a["empirical_covariance"],
         a["posterior_mean_covariance"], b["posterior_mean_covariance"]),
        ("Generating", "Empirical", "A: simple inference", "B: general inference"),
    ):
        image = draw_matrix(axis, np.asarray(matrix), title, -1.5, 1.5)
    figure.colorbar(image, ax=axes, shrink=0.75)
    figure.savefig(shared / "A_vs_B_covariance.pdf", bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for axis, matrix, title in zip(
        axes,
        (simple_generation["M"], a["posterior_mean_drift"],
         b["posterior_mean_drift"]),
        ("Generating", "A: simple inference", "B: general inference"),
    ):
        image = draw_matrix(axis, np.asarray(matrix), title, -2, 2)
    figure.colorbar(image, ax=axes, shrink=0.75)
    figure.savefig(shared / "A_vs_B_drift.pdf", bbox_inches="tight")
    plt.close(figure)

    def category_stats(summary, prefix):
        diagnostics = [item for item in summary["diagnostics"]
                       if item["quantity"].startswith(prefix)]
        return (max(item["rhat"] for item in diagnostics),
                min(item["minimum_single_chain_ess"] for item in diagnostics
                    if item["minimum_single_chain_ess"] is not None))

    rows = []
    for summary in (a, b, c):
        m_rhat, m_ess = category_stats(summary, "M_")
        s_rhat, s_ess = category_stats(summary, "Sigma_")
        rows.append((
            summary["case"], summary["inference_graph_space"],
            summary["posterior_mean_k"], summary["true_graph_posterior_probability"],
            summary["posterior_covariance_frobenius_error"],
            summary["empirical_covariance_frobenius_error"],
            summary["posterior_drift_frobenius_error"],
            m_rhat, m_ess, s_rhat, s_ess,
            summary["graph_changes_per_retained_iteration"],
            summary["chain_sampling_seconds"],
        ))
    save_csv(shared / "comparison_summary.csv",
             ("experiment", "inference_graph_space", "posterior_mean_k",
              "true_graph_posterior_probability", "posterior_covariance_frobenius_error",
              "empirical_covariance_frobenius_error", "posterior_drift_frobenius_error",
              "maximum_M_rhat", "minimum_M_single_chain_ess",
              "maximum_Sigma_rhat", "minimum_Sigma_single_chain_ess",
              "graph_changes_per_retained_iteration", "chain_sampling_seconds"),
             rows)
    save_json(shared / "comparison_summary.json", dict(
        experiments=[dict(zip(
            ("experiment", "inference_graph_space", "posterior_mean_k",
             "true_graph_posterior_probability", "posterior_covariance_frobenius_error",
             "empirical_covariance_frobenius_error", "posterior_drift_frobenius_error",
             "maximum_M_rhat", "minimum_M_single_chain_ess",
             "maximum_Sigma_rhat", "minimum_Sigma_single_chain_ess",
             "graph_changes_per_retained_iteration", "chain_sampling_seconds"), row
        )) for row in rows],
        note="A and B use the same data and hyperparameters; the graph space changes the support and normalization of the graph prior.",
    ))
    report = [
        "# Preliminary GCLM comparison",
        "",
        "Each experiment has five independent chains, 100,000 updates and 50,000 burn-in "
        "per chain. A and B use exactly the same generated observations.",
        "",
        "| Experiment | Mean k | P(true graph) | Covariance error | Drift error | "
        "Max M R-hat | Min M ESS/chain | Max Sigma R-hat | Min Sigma ESS/chain |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append(f"| {row[0]} | {row[2]:.3f} | {row[3]:.4f} | {row[4]:.3f} | "
                      f"{row[6]:.3f} | {row[7]:.3f} | {row[8]:.0f} | "
                      f"{row[9]:.3f} | {row[10]:.0f} |")
    def mean_covariance_interval_width(summary):
        interval = np.asarray(summary["posterior_covariance_interval"])
        return float((interval[1] - interval[0])[np.triu_indices(3)].mean())

    def mean_offdiagonal_drift_interval_width(summary):
        interval = np.asarray(summary["posterior_drift_interval"])
        return float((interval[1] - interval[0])[~np.eye(3, dtype=bool)].mean())

    def pooled_move_rate(summary, name):
        attempted = sum(chain["acceptance"][name]["proposed"]
                        for chain in summary["chains"])
        accepted = sum(chain["acceptance"][name]["accepted"]
                       for chain in summary["chains"])
        return accepted / attempted

    # There are C(3,2) * 2^2 = 12 simple two-edge graphs and C(6,2) = 15
    # general two-edge graphs; this is part of the graph-prior difference.
    prior_true_a = a["prior_graph_size_probabilities"][2] / 12
    prior_true_b = b["prior_graph_size_probabilities"][2] / 15
    report.extend([
        "", "For the shared simple-graph data, A and B give nearly the same "
        f"covariance errors ({a['posterior_covariance_frobenius_error']:.3f} and "
        f"{b['posterior_covariance_frobenius_error']:.3f}) and mean marginal "
        f"95% covariance interval widths ({mean_covariance_interval_width(a):.3f} "
        f"and {mean_covariance_interval_width(b):.3f}). The mean marginal "
        f"95% interval width across off-diagonal drift entries is "
        f"{mean_offdiagonal_drift_interval_width(a):.3f} under simple inference "
        f"versus {mean_offdiagonal_drift_interval_width(b):.3f} under general "
        "inference. The simple restriction narrows the drift marginals in this "
        "dataset, but its worst drift-entry ESS is still very low.",
        "", "The generating two-edge graph has prior probability "
        f"{prior_true_a:.4f} in A and {prior_true_b:.4f} in B. The corresponding "
        f"posterior probabilities are {a['true_graph_posterior_probability']:.4f} "
        f"and {b['true_graph_posterior_probability']:.4f}. Thus the graph-space "
        "comparison also changes the prior probability of the generating graph.",
        "", "For non-simple data in C, the posterior covariance mean has "
        f"Frobenius error {c['posterior_covariance_frobenius_error']:.3f} "
        f"(empirical covariance: {c['empirical_covariance_frobenius_error']:.3f}). "
        f"The generating graph has posterior probability "
        f"{c['true_graph_posterior_probability']:.4f}. The largest drift-entry "
        f"R-hat is {rows[2][7]:.3f}, versus {rows[2][9]:.3f} for covariance; "
        "drift and skew-direction cumulative means remain more variable. This "
        "supports a computational mixing concern for drift, but the generating "
        "drift need not be recovered from covariance-only observations.",
        "", "Across all updates, the death-move acceptance rates are "
        f"{pooled_move_rate(a, 'death'):.1%} (A), "
        f"{pooled_move_rate(b, 'death'):.1%} (B), and "
        f"{pooled_move_rate(c, 'death'):.1%} (C); reversal rates are "
        f"{pooled_move_rate(a, 'reversal'):.1%}, "
        f"{pooled_move_rate(b, 'reversal'):.1%}, and "
        f"{pooled_move_rate(c, 'reversal'):.1%}. These are possible proposal "
        "bottlenecks to study before the final simulation study.",
        "", "The covariance error is the Frobenius distance from the generating "
        "stationary covariance. The drift error uses the generating drift; it "
        "need not be small where drift is non-identifiable.",
        "", "Convergence flags in each diagnostics.csv use R-hat > 1.05 or a "
        "minimum approximate single-chain ESS < 100. Small R-hat alone does not "
        "establish convergence. The ESS calculation uses FFT autocorrelation "
        "with initial positive pairs; it is an approximation.",
        "", "The A/B contrast combines a change in identifiability with a change "
        "in the admissible models and truncated graph prior. It does not isolate "
        "identifiability causally.",
        "", "This is one simulated dataset per generating model. The low drift "
        "and graph ESS values, and the drift R-hat above 1.05 in C, need attention "
        "before using these runs as a final thesis simulation study. No automatic "
        "tuning or new move types were used.",
        "", "Five useful meeting figures:",
        "", "1. `shared/A_vs_B_posterior_k.pdf`",
        "2. `shared/A_vs_B_edge_inclusion.pdf`",
        "3. `shared/A_vs_B_covariance.pdf`",
        "4. `shared/A_vs_B_drift.pdf`",
        "5. `C_general_on_nonsimple/skew_cumulative_means.pdf`",
        "",
    ])
    (shared / "preliminary_summary.md").write_text("\n".join(report), encoding="utf-8")


def main():
    if not 0 <= BURN_IN < N_ITERATIONS:
        raise ValueError("Require 0 <= burn_in < n_iterations")
    if len(CHAIN_SEEDS) != 5:
        raise ValueError("This experiment requires exactly five chain seeds")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for case_name, _, _, _ in CASES:
        (OUTPUT / case_name).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "shared").mkdir(exist_ok=True)
    generation = {name: generating_setup(name) for name in DATA_SEEDS}
    for model_name, setup in generation.items():
        save_csv(OUTPUT / "shared" / f"data_{model_name}.csv",
                 ("y_1", "y_2", "y_3"), setup["data"])
    save_json(OUTPUT / "shared" / "experiment_config.json", dict(
        n_samples=N_SAMPLES, n_iterations=N_ITERATIONS, burn_in=BURN_IN,
        chain_seeds=CHAIN_SEEDS, data_seeds=DATA_SEEDS,
        prior_settings=PRIOR_SETTINGS, proposal_settings=PROPOSAL_SETTINGS,
        experiments=[dict(name=name, generating_model=model, graph_space=space,
                          initial_graphs=[[list(edge) for edge in sorted(edges)]
                                          for edges in starts])
                     for name, model, space, starts in CASES],
        generating_matrices={name: value["M"].tolist() for name, value in generation.items()},
        generating_covariances={name: value["sigma"].tolist()
                                for name, value in generation.items()},
    ))

    summaries = {}
    for case_name, model_name, graph_space, starts in CASES:
        print(f"{case_name}: model={model_name}, space={graph_space}", flush=True)
        prior_k, prior_mass = graph_prior(graph_space)
        print("  prior p(k):", dict(zip(prior_k.tolist(), prior_mass.round(4).tolist())),
              flush=True)
        stats = [run_chain(case_name, chain_id, seed, edges, graph_space,
                           generation[model_name]["data"])
                 for chain_id, (seed, edges) in enumerate(zip(CHAIN_SEEDS, starts), start=1)]
        summaries[case_name] = analyze_case(case_name, model_name, graph_space,
                                            generation[model_name], stats)
    compare_cases(summaries, generation["simple"])
    print("Saved preliminary comparison to", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
