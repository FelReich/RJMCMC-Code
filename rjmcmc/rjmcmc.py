from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Proposal:
    """A proposed state, its reverse move, and any sampled auxiliary values."""

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
        state = initial_state
        states = [state]
        moves = []

        for _ in range(n_steps):
            state, name, accepted = self.step(state)
            states.append(state)
            moves.append((name, accepted))

        return states, moves
