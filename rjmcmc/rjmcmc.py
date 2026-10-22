from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
import numpy as np


@dataclass
class Proposal:
    """A proposed state, its reverse move, and move-specific auxiliary values."""

    state: object
    reverse_move: str
    u: object = None


class BaseMove(ABC):
    """Base class for moves used by BaseRJMCMC.

    Each move constructs its own proposal and uses its own simplified
    acceptance probability, including the forward and reverse move-choice
    probabilities supplied by BaseRJMCMC.
    """

    def available(self, state):
        """Whether this move can be selected from state."""
        return True

    @abstractmethod
    def propose(self, state, rng: np.random.Generator) -> Proposal:
        """Return a new proposal state and auxiliary values without changing state."""

    @abstractmethod
    def acceptance(
        self,
        state,
        proposal: Proposal,
        forward_probability: float,
        reverse_probability: float,
        rng: np.random.Generator,
    ) -> bool:
        """Return whether the move is accepted according to acceptance its rule."""


class BaseRJMCMC:
    """Base class for the RJMCMC sampler."""

    def __init__(self, moves, move_weights, seed=None):
        if not moves:
            raise ValueError("Give at least one move")
        if not callable(move_weights):
            self._check_move_weights(moves, move_weights)

        self.moves = moves
        self.move_weights = move_weights
        self.rng = np.random.default_rng(seed)
        self.states = None
        self.move_outcomes = None

    @staticmethod
    def _check_move_weights(moves, weights):
        if set(moves) != set(weights):
            raise ValueError("Give every move exactly one weight")
        if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("Move weights must be finite and non-negative")
        if not any(weight > 0 for weight in weights.values()):
            raise ValueError("At least one move must have a positive weight")

    def move_probabilities(self, state):
        """Normalize fixed or state-dependent weights over available moves."""
        weights = (
            self.move_weights(state)
            if callable(self.move_weights)
            else self.move_weights
        )
        self._check_move_weights(self.moves, weights)
        available = {
            name: weights[name]
            for name, move in self.moves.items()
            if weights[name] > 0 and move.available(state)
        }
        if not available:
            raise ValueError("No positive-weight moves available for this state")

        total = sum(available.values())
        return {name: weight / total for name, weight in available.items()}

    def step(self, state):
        # Normalize the weights of moves available from the current state.
        probabilities = self.move_probabilities(state)
        names = list(probabilities)

        # Select one move using given probabilities.
        chosen = self.rng.choice(
            names, p=[probabilities[name] for name in names]
        )

        # Construct the proposed state and any auxiliary variables according
        # to the selected move.
        move = self.moves[chosen]
        proposal = move.propose(state, self.rng)

        # The reverse move is selected from the proposed state, where the
        # available moves and their probabilities may differ.
        reverse_move = self.moves.get(proposal.reverse_move)
        # Reject the proposal if the reverse move is not available.
        if reverse_move is None or not reverse_move.available(proposal.state):
            return state, chosen, False
        reverse_probability = self.move_probabilities(proposal.state).get(
            proposal.reverse_move, 0.0
        )
        # Reject the proposal if the probability of slecting the reverse
        # move at the proposed state is 0.
        if reverse_probability == 0:
            return state, chosen, False

        # Determine whether the proposal is accepted.
        accepted = move.acceptance(
            state,
            proposal,
            probabilities[chosen],
            reverse_probability,
            self.rng,
        )
        if not isinstance(accepted, (bool, np.bool_)):
            raise TypeError("Move.acceptance must return a boolean")
        return (proposal.state if accepted else state), chosen, bool(accepted)

    def run(self, initial_state, n_steps):
        """Run the chain and record states and move outcomes."""
        if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
            raise ValueError("n_steps must be a non-negative integer")
        state = initial_state
        states = [state]
        moves = []

        for _ in range(n_steps):
            state, name, accepted = self.step(state)
            states.append(state)
            moves.append((name, accepted))

        self.states = states
        self.move_outcomes = moves
        return states, moves

    def _check_burn_in(self, burn_in):
        if self.move_outcomes is None:
            raise RuntimeError("Run the chain before evaluating it")
        if (
            not isinstance(burn_in, (int, np.integer))
            or burn_in < 0
            or burn_in >= len(self.move_outcomes)
        ):
            raise ValueError("burn_in input is not supported")

    def _trace_values(self, statistic=None):
        if self.states is None:
            raise RuntimeError("Run the chain before evaluating it")
        # Unless a statistic is given use model indicator k.
        if statistic is None:
            if all(hasattr(state, "k") for state in self.states):
                values = [state.k for state in self.states]
            elif all(np.isscalar(state) for state in self.states):
                values = self.states
            else:
                raise ValueError("Provide a scalar statistic(state) for these states")
        else:
            values = [statistic(state) for state in self.states]
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(
                "The trace statistic must be a finite scalar for each state"
            )
        return values

    def trace_plot(self, statistic=None, burn_in=0, save_path=None, label=None):
        """Return a trace plot, using model index k unless given a statistic."""
        self._check_burn_in(burn_in)
        values = self._trace_values(statistic)

        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(9, 3.5))
        axis.plot(np.arange(len(values)), values, linewidth=0.7)
        if burn_in:
            axis.axvline(burn_in, color="tab:red", linestyle="--", label="Burn-in")
            axis.legend()
        default_label = (
            "Model index k"
            if statistic is None and hasattr(self.states[0], "k")
            else "Statistic"
        )
        axis.set(xlabel="Iteration", ylabel=label or default_label)
        figure.tight_layout()
        if save_path is not None:
            figure.savefig(save_path)
        return figure

    def acceptance_rate(self, burn_in=0):
        """Return overall and per-move acceptance rates after burn-in."""
        self._check_burn_in(burn_in)
        outcomes = self.move_outcomes[burn_in:]
        overall = sum(accepted for _, accepted in outcomes) / len(outcomes)
        by_move = {}
        for name in self.moves:
            selected = [accepted for move, accepted in outcomes if move == name]
            by_move[name] = sum(selected) / len(selected) if selected else None
        return {"overall": overall, "by_move": by_move}

    def mixing_diagnostics(self, statistic=None, burn_in=0, model_index=None):
        """Summarize a scalar trace and, when available, model visits."""
        self._check_burn_in(burn_in)
        # states[0] is the initial state; outcomes[0] produces states[1].
        values = self._trace_values(statistic)[burn_in + 1:]
        n = len(values)
        centered = values - values.mean()
        result = {
            "draws": n,
            "mean": float(values.mean()),
            "lag1_autocorrelation": None,
        }
        variance_sum = np.dot(centered, centered)
        if n > 1 and variance_sum > 0:
            result["lag1_autocorrelation"] = float(
                np.dot(centered[:-1], centered[1:]) / variance_sum
            )

        retained_states = self.states[burn_in + 1:]
        if model_index is not None or all(
            hasattr(state, "k") for state in retained_states
        ):
            labels = [
                model_index(state) if model_index is not None else state.k
                for state in retained_states
            ]
            result["model_switch_rate"] = (
                sum(a != b for a, b in zip(labels[:-1], labels[1:])) / (n - 1)
                if n > 1 else None
            )
            counts = Counter(labels)
            result["model_index_frequencies"] = {
                index: count / n for index, count in counts.items()
            }
        return result

    @staticmethod
    def two_way_anova(chains, statistic, models=None, burn_in=0, model_index=None):
        """Brooks-Giudici variance comparisons for multiple state histories.

        Each entry in chains is a history as returned by run(), including its
        initial state. burn_in counts updates, so the retained states start at
        index burn_in + 1. statistic must have the same meaning in every model.
        It should vary within models; the model index itself does not.
        By default, the model identifier is state.k. Ratios are direct quotients
        of the paired variance estimates, not square roots.
        """
        if not callable(statistic):
            raise TypeError("statistic must be a scalar function of state")
        if model_index is not None and not callable(model_index):
            raise TypeError("model_index must be a function of state")
        if not isinstance(burn_in, (int, np.integer)) or burn_in < 0:
            raise ValueError("burn_in must be a non-negative integer")

        chains = [list(chain) for chain in chains]
        if len(chains) < 2:
            raise ValueError("Two-way ANOVA requires at least two chains")
        if len({len(chain) for chain in chains}) != 1:
            raise ValueError("All chains must have equal-length state histories")
        retained = [chain[burn_in + 1:] for chain in chains]
        if not retained[0]:
            raise ValueError("At least one draw must remain after burn-in")

        observed = []
        for chain in retained:
            cells = {}
            for state in chain:
                if model_index is None:
                    if not hasattr(state, "k"):
                        raise ValueError("Provide model_index for states without k")
                    label = state.k
                else:
                    label = model_index(state)
                value = np.asarray(statistic(state), dtype=float)
                if value.ndim != 0 or not np.isfinite(value):
                    raise ValueError("statistic must return a finite scalar")
                cells.setdefault(label, []).append(float(value))
            observed.append(cells)

        if models is None:
            chosen = [
                label for label, values in observed[0].items()
                if all(len(chain.get(label, ())) >= 2 for chain in observed)
            ]
        else:
            chosen = list(dict.fromkeys(models))
        if len(chosen) < 2:
            raise ValueError("At least two eligible models are required")
        for chain in observed:
            if any(len(chain.get(label, ())) < 2 for label in chosen):
                raise ValueError(
                    "Every chosen model needs at least two draws in every chain"
                )

        n_chains = len(observed)
        n_models = len(chosen)
        cells = [
            {label: np.asarray(chain[label]) for label in chosen}
            for chain in observed
        ]
        chain_draws = [np.concatenate(list(chain.values())) for chain in cells]
        model_draws = {
            label: np.concatenate([chain[label] for chain in cells])
            for label in chosen
        }
        all_draws = np.concatenate(chain_draws)
        grand_mean = float(all_draws.mean())
        chain_means = [float(draws.mean()) for draws in chain_draws]
        model_means = {
            label: float(draws.mean()) for label, draws in model_draws.items()
        }
        cell_means = [
            {label: float(draws.mean()) for label, draws in chain.items()}
            for chain in cells
        ]

        # K(i,k), K_i, and K_k count only draws in the chosen models.
        cell_counts = [
            {label: len(draws) for label, draws in chain.items()}
            for chain in cells
        ]
        chain_counts = [len(draws) for draws in chain_draws]
        model_counts = {label: len(draws) for label, draws in model_draws.items()}

        # W_m and W_c use the corrected denominators in the supplied theory.
        v_hat = float(np.var(all_draws, ddof=1))
        b_m = sum(
            (model_means[label] - grand_mean) ** 2 for label in chosen
        ) / (n_models - 1)
        w_m = sum(
            np.sum((model_draws[label] - model_means[label]) ** 2)
            / (model_counts[label] - 1)
            for label in chosen
        ) / n_models
        w_c = sum(
            np.sum((draws - mean) ** 2) / (len(draws) - 1)
            for draws, mean in zip(chain_draws, chain_means)
        ) / n_chains
        b_m_w_c = sum(
            sum((means[label] - chain_mean) ** 2 for label in chosen)
            / (n_models - 1)
            for means, chain_mean in zip(cell_means, chain_means)
        ) / n_chains
        w_m_w_c = sum(
            np.var(chain[label], ddof=1)
            for chain in cells for label in chosen
        ) / (n_chains * n_models)

        def ratio(numerator, denominator):
            if denominator == 0:
                return float("inf") if numerator > 0 else None
            return float(numerator / denominator)

        return {
            "models": tuple(chosen),
            "n_chains": n_chains,
            "retained_draws_per_chain": len(retained[0]),
            "counts": {
                "chain_model": cell_counts,
                "chain": chain_counts,
                "model": model_counts,
            },
            "means": {
                "overall": grand_mean,
                "chain": chain_means,
                "model": model_means,
                "chain_model": cell_means,
            },
            "V_hat": v_hat,
            "B_m": float(b_m),
            "W_m": float(w_m),
            "W_c": float(w_c),
            "B_m_W_c": float(b_m_w_c),
            "W_m_W_c": float(w_m_w_c),
            "ratios": {
                "V_hat_over_W_c": ratio(v_hat, w_c),
                "W_m_over_W_m_W_c": ratio(w_m, w_m_w_c),
                "B_m_over_B_m_W_c": ratio(b_m, b_m_w_c),
            },
        }

    def evaluate(
        self, statistic=None, burn_in=0, trace_path=None, chains=None,
        models=None, model_index=None,
    ):
        """Return run summaries; use supplied histories for two-way ANOVA.

        chains, when provided, contains all histories including self.states.
        At least two chains and a cross-model scalar statistic are required.
        """
        self._check_burn_in(burn_in)
        if chains is not None and statistic is None:
            raise ValueError("Provide statistic for two-way ANOVA")
        anova = (
            self.two_way_anova(
                chains, statistic, models=models, burn_in=burn_in,
                model_index=model_index,
            )
            if chains is not None else None
        )
        return {
            "iterations": {
                "total": len(self.move_outcomes),
                "burn_in": burn_in,
                "retained": len(self.move_outcomes) - burn_in,
                "chains": anova["n_chains"] if anova is not None else 1,
            },
            "acceptance_rate": self.acceptance_rate(burn_in=burn_in),
            "mixing": self.mixing_diagnostics(
                statistic=statistic, burn_in=burn_in, model_index=model_index
            ),
            "two_way_anova": anova,
            "trace_plot": self.trace_plot(
                statistic=statistic, burn_in=burn_in, save_path=trace_path
            ),
        }
