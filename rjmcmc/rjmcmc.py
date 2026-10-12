from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Proposal:
    """A proposed state, its reverse move, and move-specific auxiliary values."""

    state: object
    reverse_move: str
    u: object = None


class BaseMove(ABC):
    """Interface for moves used by BaseRJMCMC.

    Each move constructs its own proposal and uses its own simplified
    acceptance probability, including the forward and reverse move-choice
    probabilities supplied by BaseRJMCMC.
    """

    def available(self, state):
        """Whether this move can be proposed from state."""
        return True

    @abstractmethod
    def propose(self, state, rng: np.random.Generator) -> Proposal:
        """Return a new state and auxiliary values without changing state."""

    @abstractmethod
    def acceptance(
        self,
        state,
        proposal: Proposal,
        forward_probability: float,
        reverse_probability: float,
        rng: np.random.Generator,
    ) -> bool:
        """Apply this move's acceptance rule and return whether it accepted."""


class BaseRJMCMC:
    def __init__(self, moves, move_weights, seed=None):
        if not moves or set(moves) != set(move_weights):
            raise ValueError("Give every move exactly one weight")
        if any(
            not np.isfinite(weight) or weight < 0
            for weight in move_weights.values()
        ):
            raise ValueError("Move weights must be finite and non-negative")
        if not any(weight > 0 for weight in move_weights.values()):
            raise ValueError("At least one move must have a positive weight")

        self.moves = moves
        self.move_weights = move_weights
        self.rng = np.random.default_rng(seed)
        self.states = None
        self.move_outcomes = None

    def move_probabilities(self, state):
        """Normalize weights over moves available at this state."""
        available = {
            name: self.move_weights[name]
            for name, move in self.moves.items()
            if self.move_weights[name] > 0 and move.available(state)
        }
        if not available:
            raise ValueError("No positive-weight moves available for this state")

        total = sum(available.values())
        return {name: weight / total for name, weight in available.items()}

    def step(self, state):
        # Normalize the weights of moves available from the current state.
        probabilities = self.move_probabilities(state)
        names = list(probabilities)

        # Select one move using those probabilities.
        chosen = self.rng.choice(
            names, p=[probabilities[name] for name in names]
        )

        # Construct the proposed state and any auxiliary variables.
        move = self.moves[chosen]
        proposal = move.propose(state, self.rng)

        # The reverse move is selected from the proposed state, where the
        # available moves and their probabilities may differ.
        reverse_move = self.moves.get(proposal.reverse_move)
        if (
            reverse_move is None
            or self.move_weights[proposal.reverse_move] == 0
            or not reverse_move.available(proposal.state)
        ):
            return state, chosen, False
        reverse_probability = self.move_probabilities(proposal.state).get(
            proposal.reverse_move, 0.0
        )

        # The chosen move applies its own acceptance rule, including both
        # move-choice probabilities. The sampler only records the result.
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
            raise ValueError("burn_in must leave at least one sampled step")

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
        """Return a trace figure, using model index k unless given a statistic."""
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

    def mixing_diagnostics(self, statistic=None, burn_in=0, max_lag=1000):
        """Estimate single-chain mixing."""
        self._check_burn_in(burn_in)
        if not isinstance(max_lag, (int, np.integer)) or max_lag < 1:
            raise ValueError("max_lag must be a positive integer")
        values = self._trace_values(statistic)[burn_in:]
        n = len(values)
        centered = values - values.mean()
        variance = np.dot(centered, centered) / n
        result = {
            "draws": n,
            "mean": float(values.mean()),
            "lag1_autocorrelation": None,
            "approximate_ess": None,
        }
        if n > 1 and variance > 0:
            n_lags = min(max_lag, n - 1)
            autocorrelations = np.array([
                np.dot(centered[:-lag], centered[lag:])
                / (n * variance)
                for lag in range(1, n_lags + 1)
            ])
            result["lag1_autocorrelation"] = float(autocorrelations[0])

            # Initial positive sequence: rho(0)+rho(1), rho(2)+rho(3), ...
            correlation_sum = autocorrelations[0]
            for lag in range(1, n_lags - 1, 2):
                pair = autocorrelations[lag] + autocorrelations[lag + 1]
                if pair <= 0:
                    break
                correlation_sum += pair
            result["approximate_ess"] = float(n / (1 + 2 * correlation_sum))

        if statistic is None and hasattr(self.states[0], "k"):
            result["model_switch_rate"] = (
                float(np.mean(np.diff(values) != 0)) if n > 1 else None
            )
            indices, counts = np.unique(values.astype(int), return_counts=True)
            result["model_index_frequencies"] = {
                int(index): float(count / n) for index, count in zip(indices, counts)
            }
        return result

    def evaluate(self, statistic=None, burn_in=0, trace_path=None, max_lag=1000):
        """Return the plot, acceptance rates, and mixing results for the last run."""
        return {
            "acceptance_rate": self.acceptance_rate(burn_in=burn_in),
            "mixing": self.mixing_diagnostics(
                statistic=statistic, burn_in=burn_in, max_lag=max_lag
            ),
            "trace_plot": self.trace_plot(
                statistic=statistic, burn_in=burn_in, save_path=trace_path
            ),
        }
