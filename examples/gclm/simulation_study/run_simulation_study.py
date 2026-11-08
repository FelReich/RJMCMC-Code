"""Run the GCLM simulation study specified by settings/settings.csv.

From the Code repository root:
    python3 examples/gclm/simulation_study/run_simulation_study.py --smoke-test
    python3 examples/gclm/simulation_study/run_simulation_study.py --dry-run
    python3 examples/gclm/simulation_study/run_simulation_study.py
    python3 examples/gclm/simulation_study/run_simulation_study.py --resume

Each output row is the state *after* the numbered update. Iterations are
one-based, so a burn-in of 50,000 leaves iterations 50,001 through 100,000.
Each completed chain is saved as results/<configuration_id>/chain_<number>.csv.
Vertices are zero-based; edge (i, j) means source i, target j, and M[j, i].
Bit i corresponds to the i-th edge column in lexicographic (source, target)
order. A blank b_i_j means the edge is absent, not that its coefficient is 0.

The sampler exposes attempted move, acceptance, likelihood (up to a constant),
and cached drift calculations. It does not expose log-prior, log-posterior, or proposal
log-acceptance ratios, so these are intentionally absent from the CSV.
Recording uses no additional random draws and does not change transitions.
"""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from examples.gclm.gclm import (
    GCLM, GCLMCalculations, GCLMPriorConfig, GCLMProposalConfig,
)
from examples.gclm.gclm_moves import (
    BirthMove, CoefficientMove, ContrastMove, DeathMove, MarginMove,
    MeanMove, ReversalMove,
)
from examples.gclm.state import Graph, State


HERE = Path(__file__).resolve().parent
SETTINGS_PATH = HERE / "settings" / "settings.csv"
RESULTS_DIR = HERE / "results"
MOVE_CLASSES = {
    "mean": MeanMove,
    "contrast": ContrastMove,
    "margin": MarginMove,
    "coefficient": CoefficientMove,
    "reversal": ReversalMove,
    "birth": BirthMove,
    "death": DeathMove,
}


def load_settings():
    """Read the finalized numerical settings, including the true matrices."""
    with SETTINGS_PATH.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["section", "key", "value_json"]:
            raise ValueError("settings.csv must have section,key,value_json columns")
        settings = {}
        configurations = []
        for row in reader:
            identifier = (row["section"], row["key"])
            if identifier in settings:
                raise ValueError(f"Duplicate settings entry: {identifier}")
            value = json.loads(row["value_json"])
            settings[identifier] = value
            if row["section"] == "configuration":
                configurations.append((row["key"], value))

    p = settings["dimensions", "p"]
    mean = np.asarray(settings["dimensions", "generating_mean"], dtype=float)
    diffusion = np.asarray(settings["dimensions", "diffusion_covariance_C"], dtype=float)
    if not isinstance(p, int) or p < 2 or mean.shape != (p,) or not np.all(np.isfinite(mean)):
        raise ValueError("Invalid dimension or generating mean in settings.csv")
    # GCLMCalculations currently implements this diffusion covariance only.
    if diffusion.shape != (p, p) or not np.array_equal(diffusion, 2 * np.eye(p)):
        raise ValueError("The existing GCLM sampler requires C = 2 I_p")

    sample_sizes = settings["dimensions", "sample_sizes"]
    graph_spaces = settings["dimensions", "inference_graph_spaces"]
    generating_models = {}
    for (section, key), value in settings.items():
        if section != "generating_model":
            continue
        covariance = np.asarray(value["Sigma"], dtype=float)
        drift = np.asarray(value["M"], dtype=float)
        if (covariance.shape != (p, p) or drift.shape != (p, p)
                or not np.all(np.isfinite(covariance))
                or not np.all(np.isfinite(drift))
                or not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0)
                or np.linalg.eigvalsh(covariance).min() <= 0):
            raise ValueError(f"Invalid generating matrices for {key}")
        graph = Graph(p, frozenset(map(tuple, value["ordered_edges_zero_based"])))
        matrix_edges = {(i, j) for i in range(p) for j in range(p)
                        if i != j and drift[j, i] != 0}
        if (graph.k != value["k"] or not graph.is_simple
                or matrix_edges != graph.edges
                or np.linalg.eigvals(drift).real.max() >= 0
                or not np.allclose(drift @ covariance + covariance @ drift.T + diffusion,
                                   0, atol=1e-10, rtol=0)):
            raise ValueError(f"Inconsistent generating graph, drift, or covariance for {key}")
        generating_models[key] = covariance

    run = settings["mcmc", "run_settings"]
    prior = settings["prior", "hyperparameters"]
    proposal = settings["proposal", "parameters_and_base_weights"]
    move_order = settings["proposal", "move_order"]
    if set(move_order) != set(MOVE_CLASSES) or len(move_order) != len(MOVE_CLASSES):
        raise ValueError("The recorded move order does not match the GCLM moves")
    if not configurations or len(configurations) != run["total_configurations"]:
        raise ValueError("The configuration count does not match the run settings")
    if run["iterations_per_chain"] <= run["burn_in"] or run["burn_in"] < 0:
        raise ValueError("Invalid iteration count or burn-in in settings.csv")
    if run["retained_per_chain"] != run["iterations_per_chain"] - run["burn_in"]:
        raise ValueError("The retained iteration count does not match settings.csv")

    datasets = {}
    for (section, key), value in settings.items():
        if section == "dataset":
            datasets[key] = value
    tasks = []
    seen_chain_seeds = set()
    for config_id, config in configurations:
        if not config_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in config_id):
            raise ValueError(f"Unsafe configuration ID: {config_id!r}")
        model = config["generating_model"]
        n = config["n"]
        space = config["inference_graph_space"]
        if model not in generating_models or model not in datasets:
            raise ValueError(f"Unknown generating model for {config_id}")
        if n not in sample_sizes or space not in graph_spaces:
            raise ValueError(f"Unexpected sample size or graph space for {config_id}")
        dataset = datasets[model]
        if (config["dataset_seed"] != dataset["seed"]
                or config["dataset_row_slice"] != [0, n]
                or n > dataset["maximum_sample_size"]):
            raise ValueError(f"Inconsistent dataset specification for {config_id}")
        seeds = config["chain_seeds"]
        if len(seeds) != run["chains_per_configuration"]:
            raise ValueError(f"Wrong number of chain seeds for {config_id}")
        for chain_number, seed in enumerate(seeds, start=1):
            if not isinstance(seed, int) or seed in seen_chain_seeds:
                raise ValueError("Chain seeds must be distinct integers")
            seen_chain_seeds.add(seed)
            tasks.append((config_id, config, chain_number, seed))
    if len(tasks) != run["total_chains"]:
        raise ValueError("The chain count does not match settings.csv")

    return settings, tasks, generating_models, datasets, mean, prior, proposal, move_order


def edge_order(p):
    """Order of bitmask bits, edge indicators, and coefficient columns."""
    return tuple((i, j) for i in range(p) for j in range(p) if i != j)


def columns(p):
    edges = edge_order(p)
    names = [
        "configuration_id", "generating_model", "n", "inference_graph_space",
        "chain_number", "chain_seed", "dataset_seed", "iteration",
        "move_type", "accepted", "graph_changed", "k", "graph_bitmask",
        "graph_edges_zero_based",
    ]
    names += [f"edge_{i}_{j}" for i, j in edges]
    names += [f"mu_{i}" for i in range(p)]
    names += [f"r_{i}" for i in range(p - 1)]
    names += ["eta"]
    names += [f"b_{i}_{j}" for i, j in edges]
    names += [
        "phi_H", "stability_shift", "phi_M",
        "relative_lyapunov_residual", "log_likelihood",
    ]
    names += [f"M_{i}_{j}" for i in range(p) for j in range(p)]
    names += [f"Sigma_{i}_{j}" for i in range(p) for j in range(p)]
    return names


def number(value):
    """Round-trip a binary64 number with 17 significant decimal digits."""
    return format(float(value), ".17g")


def make_row(task, iteration, move, accepted, previous_graph,
             state, calculations, edges):
    config_id, config, chain_number, seed = task
    graph = state.graph
    indicators = [int(edge in graph.edges) for edge in edges]
    bitmask = sum(bit << index for index, bit in enumerate(indicators))
    coefficients = state.coefficients_by_edge
    calculation = calculations.drift(state)  # Current cache: no new Lyapunov solve.
    shift = calculation.spectral_abscissa + state.eta
    phi_m = calculation.spectral_abscissa - shift  # Eigenvalue-shift identity.
    row = [
        config_id, config["generating_model"], config["n"],
        config["inference_graph_space"], chain_number, seed,
        config["dataset_seed"], iteration, move, int(accepted),
        int(graph != previous_graph), graph.k, bitmask,
        json.dumps([list(edge) for edge in graph.ordered_edges], separators=(",", ":")),
    ]
    row += indicators
    row += [number(value) for value in state.mu]
    row += [number(value) for value in state.r]
    row += [number(state.eta)]
    row += [number(coefficients[edge]) if edge in coefficients else "" for edge in edges]
    row += [
        number(calculation.spectral_abscissa), number(shift), number(phi_m),
        number(calculation.relative_lyapunov_residual),
        number(calculations.log_likelihood(state)),
    ]
    row += [number(value) for value in calculation.M.flat]
    row += [number(value) for value in calculation.covariance.flat]
    return row


def generate_datasets(models, datasets, mean, chosen_models):
    """Draw one maximum-size dataset per generating graph, then reuse prefixes."""
    observations = {}
    for model in chosen_models:
        spec = datasets[model]
        observations[model] = np.random.default_rng(spec["seed"]).multivariate_normal(
            mean, models[model], size=spec["maximum_sample_size"]
        )
    return observations


def run_chain(task, data, prior_values, proposal_values, move_order,
              n_iterations, burn_in, output, fieldnames, edges):
    """Run one chain and stream every post-burn-in state to an open CSV stream."""
    config_id, config, chain_number, seed = task
    prior = GCLMPriorConfig(graph_space=config["inference_graph_space"], **prior_values)
    proposal = GCLMProposalConfig(**proposal_values)
    calculations = GCLMCalculations(data)
    moves = {
        name: MOVE_CLASSES[name](calculations, prior, proposal)
        for name in move_order
    }
    sampler = GCLM(moves, calculations, prior, proposal, seed=seed)
    retained = 0
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(fieldnames)
    for iteration in range(1, n_iterations + 1):
        previous_graph = sampler.state.graph
        state, move, accepted = sampler.step()
        if iteration <= burn_in:
            continue
        if config["inference_graph_space"] == "simple" and not state.graph.is_simple:
            raise RuntimeError("The sampler produced a graph outside the simple space")
        writer.writerow(make_row(
            task, iteration, move, accepted, previous_graph,
            state, calculations, edges,
        ))
        retained += 1
    if retained != n_iterations - burn_in:
        raise RuntimeError(f"Wrong retained row count for {config_id} chain {chain_number}")
    return retained


def verify_completed_chain(path, task, fieldnames, n_iterations, burn_in):
    """A completed chain must have the full header and consecutive iterations."""
    config_id, config, chain_number, seed = task
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != fieldnames:
            raise ValueError(f"Wrong columns in completed chain file {path}")
        count = 0
        for row in reader:
            expected_iteration = burn_in + count + 1
            if (None in row or any(value is None for value in row.values())
                    or row["configuration_id"] != config_id
                    or row["generating_model"] != config["generating_model"]
                    or row["n"] != str(config["n"])
                    or row["inference_graph_space"] != config["inference_graph_space"]
                    or row["chain_number"] != str(chain_number)
                    or row["chain_seed"] != str(seed)
                    or row["dataset_seed"] != str(config["dataset_seed"])
                    or row["iteration"] != str(expected_iteration)):
                raise ValueError(f"Invalid row in completed chain file {path}")
            count += 1
    if count != n_iterations - burn_in:
        raise ValueError(f"Incomplete chain file {path}: {count} retained rows")


def chain_file(task):
    config_id, _, chain_number, _ = task
    return RESULTS_DIR / config_id / f"chain_{chain_number}.csv"


def prepare_results(settings_hash, fieldnames, run, resume):
    """Tie resumable chain files to the exact settings and output schema."""
    manifest_path = RESULTS_DIR / "manifest.json"
    manifest = {
        "settings_sha256": settings_hash,
        "columns": fieldnames,
        "iterations_per_chain": run["iterations_per_chain"],
        "burn_in": run["burn_in"],
    }
    if RESULTS_DIR.exists() and not RESULTS_DIR.is_dir():
        raise ValueError(f"Results path is not a directory: {RESULTS_DIR}")
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"Results already exist in {RESULTS_DIR}; use --resume to continue"
            )
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing chain files do not match the current settings/schema")
    elif RESULTS_DIR.exists() and any(
        path.is_file() and path.name != ".DS_Store" for path in RESULTS_DIR.rglob("*")
    ):
        raise ValueError("Result files exist without a manifest; cannot continue safely")
    RESULTS_DIR.mkdir(exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def verify_smoke_csv(text, task, data, fieldnames, edges):
    """Parse the in-memory CSV and reconstruct every retained state."""
    config_id, config, chain_number, seed = task
    p = data.shape[1]
    calculations = GCLMCalculations(data)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != fieldnames:
        raise ValueError("Smoke test produced the wrong CSV columns")

    count = 0
    max_residual = 0.0
    for row in reader:
        count += 1
        if (None in row or any(value is None for value in row.values())
                or row["configuration_id"] != config_id
                or row["generating_model"] != config["generating_model"]
                or row["n"] != str(config["n"])
                or row["inference_graph_space"] != "simple"
                or row["chain_number"] != str(chain_number)
                or row["chain_seed"] != str(seed)
                or row["dataset_seed"] != str(config["dataset_seed"])
                or row["iteration"] != str(50 + count)):
            raise ValueError(f"Invalid smoke-test CSV row {count}")

        serialized_edges = tuple(tuple(edge) for edge in json.loads(row["graph_edges_zero_based"]))
        graph = Graph(p, frozenset(serialized_edges))
        if (serialized_edges != graph.ordered_edges or not graph.is_simple
                or int(row["k"]) != graph.k):
            raise ValueError(f"Invalid graph in smoke-test row {count}")
        indicators = [int(row[f"edge_{i}_{j}"]) for i, j in edges]
        bitmask = sum(bit << index for index, bit in enumerate(indicators))
        if (indicators != [int(edge in graph.edges) for edge in edges]
                or int(row["graph_bitmask"]) != bitmask):
            raise ValueError(f"Graph columns disagree in smoke-test row {count}")

        coefficients = {}
        for i, j in edges:
            value = row[f"b_{i}_{j}"]
            if (i, j) in graph.edges:
                coefficients[i, j] = float(value)
            elif value != "":
                raise ValueError(f"Absent edge has a coefficient in smoke-test row {count}")
        state = State.from_coefficients(
            graph,
            np.array([float(row[f"mu_{i}"]) for i in range(p)]),
            np.array([float(row[f"r_{i}"]) for i in range(p - 1)]),
            float(row["eta"]), coefficients,
        )
        state.validate_for_model(p, "simple")
        calculation = calculations.drift(state)
        stored_m = np.array([float(row[f"M_{i}_{j}"])
                             for i in range(p) for j in range(p)]).reshape(p, p)
        stored_sigma = np.array([float(row[f"Sigma_{i}_{j}"])
                                 for i in range(p) for j in range(p)]).reshape(p, p)
        np.testing.assert_allclose(stored_m, calculation.M, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(stored_sigma, calculation.covariance,
                                   rtol=1e-12, atol=1e-12)
        for name, expected in (
            ("phi_H", calculation.spectral_abscissa),
            ("stability_shift", calculation.spectral_abscissa + state.eta),
            ("phi_M", -state.eta),
            ("relative_lyapunov_residual", calculation.relative_lyapunov_residual),
            ("log_likelihood", calculations.log_likelihood(state)),
        ):
            if not np.isclose(float(row[name]), expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"{name} disagrees in smoke-test row {count}")
        if (np.linalg.eigvals(stored_m).real.max() >= 0
                or not np.isclose(np.linalg.eigvals(stored_m).real.max(),
                                  -state.eta, rtol=1e-10, atol=1e-10)
                or not np.allclose(stored_sigma, stored_sigma.T, rtol=0, atol=1e-10)
                or np.linalg.eigvalsh(stored_sigma).min() <= 0):
            raise ValueError(f"Unstable drift or invalid covariance in smoke-test row {count}")
        lyapunov_residual = np.linalg.norm(
            stored_m @ stored_sigma + stored_sigma @ stored_m.T + 2 * np.eye(p),
            ord="fro",
        )
        max_residual = max(max_residual, lyapunov_residual)
        if lyapunov_residual > 1e-8:
            raise ValueError(f"Lyapunov equation fails in smoke-test row {count}")
    if count != 50:
        raise ValueError(f"Smoke test recorded {count} rows; expected 50")
    return max_residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one saved chain for 100 iterations, with 50 burn-in")
    parser.add_argument("--dry-run", action="store_true",
                        help="List all configurations and seeds without running MCMC")
    parser.add_argument("--resume", action="store_true",
                        help="Continue completed chain files from an interrupted full study")
    args = parser.parse_args()
    if args.smoke_test and args.dry_run:
        parser.error("Choose either --smoke-test or --dry-run")
    if args.resume and (args.smoke_test or args.dry_run):
        parser.error("--resume applies only to the full study")

    settings, tasks, models, datasets, mean, prior, proposal, move_order = load_settings()
    run = settings["mcmc", "run_settings"]
    p = settings["dimensions", "p"]
    fieldnames = columns(p)
    edges = edge_order(p)
    if args.dry_run:
        print(f"{len(tasks)} chains across {run['total_configurations']} configurations")
        for config_id, config, chain_number, seed in tasks:
            print(f"{config_id}: chain {chain_number}, seed {seed}, dataset seed {config['dataset_seed']}")
        return

    start_time = perf_counter()
    if args.smoke_test:
        task = next((task for task in tasks if task[0] == "G5_n200_simple" and task[2] == 1), None)
        if task is None:
            raise ValueError("The smoke-test configuration G5_n200_simple is absent")
        config = task[1]
        observations = generate_datasets(models, datasets, mean,
                                         {config["generating_model"]})
        data = observations[config["generating_model"]][:config["n"]]
        output = io.StringIO(newline="")
        count = run_chain(task, data, prior, proposal, move_order, 100, 50,
                          output, fieldnames, edges)
        max_residual = verify_smoke_csv(output.getvalue(), task, data, fieldnames, edges)
        print(f"Smoke test passed: {task[0]}, chain 1, seed {task[3]}, "
              f"dataset seed {config['dataset_seed']}; {count} rows (iterations 51–100), "
              f"{len(fieldnames)} columns; all states reconstructed, simple, stable, "
              f"and positive definite; maximum Lyapunov residual {max_residual:.2e}; "
              f"{perf_counter() - start_time:.1f}s. No files written.", flush=True)
        return

    settings_hash = hashlib.sha256(SETTINGS_PATH.read_bytes()).hexdigest()
    prepare_results(settings_hash, fieldnames, run, args.resume)
    observations = generate_datasets(models, datasets, mean, set(datasets))
    completed = 0
    saved = 0
    for task in tasks:
        config_id, config, chain_number, seed = task
        done = chain_file(task)
        if done.exists():
            verify_completed_chain(done, task, fieldnames,
                                   run["iterations_per_chain"], run["burn_in"])
            completed += 1
            print(f"[{completed}/{len(tasks)}] Resumed {config_id} chain {chain_number}", flush=True)
            continue
        done.parent.mkdir(exist_ok=True)
        partial = done.with_name(done.name + ".partial")
        data = observations[config["generating_model"]][:config["n"]]
        chain_start = perf_counter()
        print(f"[{completed + 1}/{len(tasks)}] Running {config_id} chain {chain_number} "
              f"(seed {seed})", flush=True)
        with partial.open("w", newline="", encoding="utf-8") as output:
            count = run_chain(task, data, prior, proposal, move_order,
                              run["iterations_per_chain"], run["burn_in"],
                              output, fieldnames, edges)
            output.flush()
            os.fsync(output.fileno())
        if done.exists():
            raise FileExistsError(f"Refusing to overwrite completed chain {done}")
        os.replace(partial, done)
        completed += 1
        saved += 1
        print(f"[{completed}/{len(tasks)}] Completed {config_id} chain {chain_number}: "
              f"{count} retained iterations, {perf_counter() - chain_start:.1f}s; "
              f"elapsed {perf_counter() - start_time:.1f}s", flush=True)
        print(f"Saved: {done.relative_to(HERE)}", flush=True)
    print(f"Study complete: {run['total_configurations']} configurations, "
          f"{completed} chains, {completed * run['retained_per_chain']} retained draws "
          f"({saved} new files) in {perf_counter() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
