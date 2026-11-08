"""Generate the self-contained settings.csv for the GCLM simulation study.

Run from the Code repository root:
    python3 examples/gclm/simulation_study/settings/settings.py

The CSV has columns section,key,value_json. Parse each value_json cell with
json.loads; no later script needs to import or execute this module. Re-running
does not replace the CSV unless --overwrite is supplied explicitly.
"""

import argparse
import csv
import json
from math import factorial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.linalg import solve_continuous_lyapunov

from examples.gclm.gclm import (
    GCLM, GCLMCalculations, GCLMPriorConfig, GCLMProposalConfig,
)
from examples.gclm.state import Graph, State


P = 5
GENERATING_SEED = 2026
TRUE_TAU = 2.0
SAMPLE_SIZES = (50, 200, 800)
TRUE_EDGE_COUNTS = (2, 5, 8)
GRAPH_SPACES = ("simple", "general")
N_CHAINS = 5
N_ITERATIONS = 100_000
BURN_IN = 50_000
DATASET_SEEDS = {2: 2_026_002, 5: 2_026_005, 8: 2_026_008}
CHAIN_SEED_START = 202_610_001
OUTPUT = Path(__file__).with_name("settings.csv")

# Edges use zero-based (source, target) indices, as in Graph and State.
EDGES = {
    2: frozenset({(0, 1), (0, 2)}),
    5: frozenset({(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)}),
    8: frozenset({
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3),
        (0, 4), (1, 4), (2, 3),
    }),
}
PRIOR = dict(
    lambda_p=5.0, s_r=1.0, mu_eta=0.0, s_eta=0.5,
    tau=2.0, m_0=0.0, s_mu=5.0,
)
PROPOSAL = dict(
    sigma_r=1.25, sigma_eta=0.125, sigma_b=0.5, gamma=0.5,
    move_weights={
        "mean": 1.0, "contrast": 1.0, "margin": 1.0,
        "coefficient": 1.0, "reversal": 5.0,
        "birth": 10.0, "death": 10.0,
    },
)


def graph_size_prior(graph_space):
    """The graph-size prior used by GCLM, normalized on its support."""
    max_edges = P * (P - 1) // (2 if graph_space == "simple" else 1)
    unnormalized = [PRIOR["lambda_p"] ** k / factorial(k)
                    for k in range(max_edges + 1)]
    total = sum(unnormalized)
    return dict(support=list(range(max_edges + 1)),
                probabilities=[weight / total for weight in unnormalized],
                conditional_graph_prior="uniform among admissible graphs with k edges")


def validate_calculation(graph, state, calculation, eta, Q, coefficient_by_edge):
    """Check the stored generating model against the thesis parametrization."""
    H, M, Sigma = calculation.H, calculation.M, calculation.covariance
    if graph.k not in TRUE_EDGE_COUNTS or not graph.is_simple:
        raise ValueError("Generating graph must be one of the specified simple DAGs")
    if not np.allclose(np.diag(H), Q @ state.r, atol=1e-12, rtol=0):
        raise ValueError("H does not use the sampler's contrast parametrization")
    if not np.allclose(M, H - (calculation.spectral_abscissa + eta) * np.eye(P),
                       atol=1e-12, rtol=0):
        raise ValueError("Drift parametrization does not match GCLM")
    observed_edges = {(source, target) for source in range(P) for target in range(P)
                      if source != target and M[target, source] != 0}
    if observed_edges != graph.edges:
        raise ValueError("Drift sparsity does not match the generating graph")
    if not np.isclose(np.max(np.linalg.eigvals(M).real), -eta, atol=1e-12, rtol=0):
        raise ValueError("Generating drift has the wrong spectral abscissa")
    for edge in graph.ordered_edges:
        source, target = edge
        if M[target, source] != coefficient_by_edge[edge]:
            raise ValueError("A shared edge coefficient changed")
    if not np.allclose(Sigma, Sigma.T, atol=1e-12, rtol=0):
        raise ValueError("Stationary covariance is not symmetric")
    if np.linalg.eigvalsh(Sigma).min() <= 0:
        raise ValueError("Stationary covariance is not positive definite")
    if not np.allclose(Sigma, solve_continuous_lyapunov(M, -2 * np.eye(P)),
                       atol=1e-12, rtol=0):
        raise ValueError("Stored covariance differs from SciPy's Lyapunov solution")
    residual = M @ Sigma + Sigma @ M.T + 2 * np.eye(P)
    relative_residual = float(np.linalg.norm(residual, ord="fro") / (
        np.linalg.norm(M @ Sigma, ord="fro")
        + np.linalg.norm(Sigma @ M.T, ord="fro")
        + np.linalg.norm(2 * np.eye(P), ord="fro")
    ))
    if not np.isfinite(relative_residual) or relative_residual > 1e-10:
        raise ValueError("Generating Lyapunov residual exceeds tolerance")
    return relative_residual


def make_rows():
    """Sample true parameters once and return every setting as a CSV row."""
    if not EDGES[2] < EDGES[5] < EDGES[8]:
        raise ValueError("Generating edge sets must be strictly nested")
    for space in GRAPH_SPACES:
        GCLMPriorConfig(graph_space=space, **PRIOR)
    GCLMProposalConfig(**PROPOSAL)
    if set(PROPOSAL["move_weights"]) != set(GCLM.MOVE_NAMES):
        raise ValueError("Proposal weights do not cover the existing moves")

    rng = np.random.default_rng(GENERATING_SEED)
    r_star = rng.standard_normal(P - 1)
    full_graph = Graph(P, EDGES[8])
    b_full = TRUE_TAU * rng.standard_normal(full_graph.k)
    eta_star = float(np.exp(rng.normal(0.0, PRIOR["s_eta"])))
    coefficient_by_edge = dict(zip(full_graph.ordered_edges, b_full))
    Q = GCLMCalculations._contrast_basis(P)
    if not np.allclose(Q.T @ Q, np.eye(P - 1), atol=1e-12, rtol=0):
        raise ValueError("Contrast basis is not orthonormal")
    if not np.allclose(Q.T @ np.ones(P), 0, atol=1e-12, rtol=0):
        raise ValueError("Contrast basis is not orthogonal to the all-ones vector")

    rows = []
    def add(section, key, value):
        rows.append((section, key, value))

    add("metadata", "schema_version", 1)
    add("metadata", "value_format", "JSON in value_json; zero-based directed edge [source,target]")
    add("metadata", "edge_matrix_convention", "source i -> target j means M[j,i]")
    add("metadata", "array_dtype", "float64")
    add("metadata", "numpy_version", np.__version__)
    add("metadata", "scipy_version", scipy.__version__)
    add("dimensions", "p", P)
    add("dimensions", "diffusion_covariance_C", (2 * np.eye(P)).tolist())
    add("dimensions", "generating_mean", np.zeros(P).tolist())
    add("dimensions", "sample_sizes", list(SAMPLE_SIZES))
    add("dimensions", "true_edge_counts", list(TRUE_EDGE_COUNTS))
    add("dimensions", "inference_graph_spaces", list(GRAPH_SPACES))
    add("prior", "hyperparameters", PRIOR)
    add("prior", "mean_covariance", (PRIOR["s_mu"] ** 2 * np.eye(P)).tolist())
    add("prior", "graph_size_by_space", {space: graph_size_prior(space)
                                         for space in GRAPH_SPACES})
    add("proposal", "parameters_and_base_weights", PROPOSAL)
    add("proposal", "move_order", list(GCLM.MOVE_NAMES))
    add("proposal", "selection_rule", "Use GCLM._gclm_move_weights and normalize among available moves")
    add("mcmc", "run_settings", dict(
        chains_per_configuration=N_CHAINS, iterations_per_chain=N_ITERATIONS,
        burn_in=BURN_IN, retained_per_chain=N_ITERATIONS - BURN_IN,
        datasets_per_generating_graph=1,
        total_configurations=len(SAMPLE_SIZES) * len(TRUE_EDGE_COUNTS) * len(GRAPH_SPACES),
        total_chains=len(SAMPLE_SIZES) * len(TRUE_EDGE_COUNTS) * len(GRAPH_SPACES) * N_CHAINS,
    ))
    add("generating_parameters", "seed", GENERATING_SEED)
    add("generating_parameters", "true_tau", TRUE_TAU)
    add("generating_parameters", "r_star", r_star.tolist())
    add("generating_parameters", "b_G8_ordered_edges", [list(edge) for edge in full_graph.ordered_edges])
    add("generating_parameters", "b_G8", b_full.tolist())
    add("generating_parameters", "eta_star", eta_star)
    add("generating_parameters", "contrast_basis_Q", Q.tolist())

    # The placeholder is only needed to call the sampler's existing drift
    # calculation. Study observations are generated later from settings.csv.
    calculations = GCLMCalculations(np.zeros((1, P)))
    model_diagonals = []
    for k in TRUE_EDGE_COUNTS:
        graph = Graph(P, EDGES[k])
        graph.validate_space("simple")
        state = State.from_coefficients(
            graph, np.zeros(P), r_star, eta_star,
            {edge: coefficient_by_edge[edge] for edge in graph.ordered_edges},
        )
        result = calculations.drift(state)
        residual = validate_calculation(graph, state, result, eta_star,
                                        Q, coefficient_by_edge)
        model_diagonals.append(np.diag(result.M).copy())
        add("generating_model", f"G{k}", dict(
            k=k,
            ordered_edges_zero_based=[list(edge) for edge in graph.ordered_edges],
            ordered_edges_one_based=[[i + 1, j + 1] for i, j in graph.ordered_edges],
            coefficients_in_order=state.b.tolist(),
            H=result.H.tolist(), M=result.M.tolist(),
            Sigma=result.covariance.tolist(),
            spectral_abscissa_H=result.spectral_abscissa,
            spectral_abscissa_M=float(np.max(np.linalg.eigvals(result.M).real)),
            relative_lyapunov_residual=residual,
        ))
    if not all(np.allclose(diagonal, model_diagonals[0], atol=1e-12, rtol=0)
               for diagonal in model_diagonals[1:]):
        raise ValueError("Nested DAGs did not produce identical drift diagonals")

    all_seeds = [GENERATING_SEED, *DATASET_SEEDS.values()]
    for k in TRUE_EDGE_COUNTS:
        add("dataset", f"G{k}", dict(
            generating_model=f"G{k}", seed=DATASET_SEEDS[k],
            maximum_sample_size=max(SAMPLE_SIZES),
            generation_rule="default_rng(seed).multivariate_normal(generating_mean, Sigma_Gk, size=800)",
            sample_size_rule="Use the first n rows of the same 800-observation dataset",
        ))
    configuration_index = 0
    for n in SAMPLE_SIZES:
        for k in TRUE_EDGE_COUNTS:
            for space in GRAPH_SPACES:
                seeds = [CHAIN_SEED_START + N_CHAINS * configuration_index + chain
                         for chain in range(N_CHAINS)]
                all_seeds.extend(seeds)
                add("configuration", f"G{k}_n{n}_{space}", dict(
                    generating_model=f"G{k}", n=n, inference_graph_space=space,
                    dataset_seed=DATASET_SEEDS[k], dataset_row_slice=[0, n],
                    chain_seeds=seeds,
                ))
                configuration_index += 1
    if len(all_seeds) != 1 + len(DATASET_SEEDS) + 90 or len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Generating, dataset, and chain seeds must all be distinct")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace an existing settings.csv")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.overwrite:
        parser.error(f"{OUTPUT} already exists; use --overwrite to regenerate it")
    rows = make_rows()
    with OUTPUT.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(("section", "key", "value_json"))
        for section, key, value in rows:
            writer.writerow((section, key, json.dumps(value, ensure_ascii=False,
                                                     allow_nan=False, separators=(",", ":"))))
    print(f"Wrote {OUTPUT} with {len(rows)} settings rows and 18 configurations (90 chain seeds).")


if __name__ == "__main__":
    main()
