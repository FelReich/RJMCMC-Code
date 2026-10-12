# Multiple change-point model for the coal-mining disaster data.

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Resolve the data path from this file, independent of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "coal_disasters.csv"
END_DAY = 40_907

from rjmcmc.rjmcmc import BaseRJMCMC


@dataclass(eq=False)
class State:
    """One model state: k change points and k + 1 intensities."""

    k: int
    change_points: np.ndarray
    intensities: np.ndarray

    def __post_init__(self):
        if not isinstance(self.k, (int, np.integer)) or self.k < 0:
            raise ValueError("k must be a non-negative integer")
        self.change_points = np.asarray(self.change_points, dtype=float).copy()
        self.intensities = np.asarray(self.intensities, dtype=float).copy()
        if self.change_points.ndim != 1 or self.intensities.ndim != 1:
            raise ValueError("Change points and intensities must be one-dimensional")
        if len(self.change_points) != self.k:
            raise ValueError("The number of change points must equal k")
        if len(self.intensities) != self.k + 1:
            raise ValueError("There must be k + 1 intensities")
        if not np.all(np.isfinite(self.change_points)) or np.any(
            np.diff(self.change_points) <= 0
        ):
            raise ValueError("Change points must be finite and strictly increasing")
        if not np.all(np.isfinite(self.intensities)) or np.any(self.intensities <= 0):
            raise ValueError("Intensities must be finite and positive")


class ChangePoints:
    """Draw and edit ordered change-point locations."""

    def __init__(self, L, n, rng=None):
        self.L = L
        self.rng = rng if rng is not None else np.random.default_rng()
        self.loc = self.init_sample(n)

    def init_sample(self, n):
        if n == 0:
            return np.array([])

        s = np.sort(self.rng.uniform(0, self.L, 2 * n + 1))
        return s[1::2]

    def add_sample(self, n):
        s = self.rng.uniform(0, self.L, n)
        self.loc = np.sort(np.concatenate((self.loc, s)))

    def remove_sample(self, idx):
        self.loc = np.delete(self.loc, idx)

    def n(self):
        return len(self.loc)


def load_disaster_days(path: Path = DATA_FILE) -> np.ndarray:
    """Read event days from the shared data file in ``Code/data``."""
    days = np.loadtxt(
        path, delimiter=",", skiprows=1, usecols=0, dtype=int, ndmin=1
    )
    if days.size == 0 or np.any(days < 0) or np.any(days >= END_DAY):
        raise ValueError(f"Invalid disaster days in {path}")
    if np.any(np.diff(days) < 0):
        raise ValueError(f"Disaster days must be ordered in {path}")
    return days


class MultipleChangePointModel(BaseRJMCMC):
    """RJMCMC sampler with Green's model-dependent move probabilities."""

    def __init__(
        self, moves, alpha, beta, param, kmax, L,
        initial_state: State | None = None, seed=None, move_weights=None,
    ):
        if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(beta) or beta <= 0:
            raise ValueError("Gamma prior parameters alpha and beta must be positive")
        if not np.isfinite(param) or param < 0:
            raise ValueError("Poisson prior parameter must be non-negative")
        if not isinstance(kmax, (int, np.integer)) or kmax < 0:
            raise ValueError("kmax must be a non-negative integer")
        if not np.isfinite(L) or L <= 0:
            raise ValueError("Observation length L must be positive")
        self.alpha = alpha
        self.beta = beta
        self.param = param
        self.kmax = kmax
        self.L = L

        if move_weights is None:
            expected = {"height", "position", "birth", "death"}
            if set(moves) != expected:
                raise ValueError("Green's move probabilities require all four moves")

            # For a truncated Poisson prior, p(k+1)/p(k) = param/(k+1).
            raw_birth = np.zeros(kmax + 1)
            raw_death = np.zeros(kmax + 1)
            for k in range(kmax):
                raw_birth[k] = min(1.0, param / (k + 1))
            for k in range(1, kmax + 1):
                raw_death[k] = min(1.0, k / param) if param > 0 else 1.0

            # Largest common c satisfying b_k + d_k <= 0.9 at every k.
            largest_sum = np.max(raw_birth + raw_death)
            self.c = 0.9 / largest_sum if largest_sum > 0 else 0.0
            self._birth_probabilities = self.c * raw_birth
            self._death_probabilities = self.c * raw_death
            move_weights = self._green_move_weights

        super().__init__(moves, move_weights, seed)

        if initial_state is None:
            initial_state = State(
                k=0,
                change_points=np.empty(0),
                intensities=self.sample_gamma_prior(),
            )
        if not isinstance(initial_state, State):
            raise TypeError("initial_state must be a State")
        if (
            initial_state.k > kmax
            or np.any(initial_state.change_points <= 0)
            or np.any(initial_state.change_points >= L)
        ):
            raise ValueError(
                "Initial change points must lie in (0, L), with at most kmax points"
            )
        self.state = initial_state

    def _green_move_weights(self, state):
        """Green's birth/death probabilities and the remaining within-model mass."""
        k = state.k
        if not 0 <= k <= self.kmax:
            raise ValueError("State k must lie between 0 and kmax")
        birth = float(self._birth_probabilities[k])
        death = float(self._death_probabilities[k])
        remaining = 1.0 - birth - death
        if k == 0:
            return {
                "height": remaining,
                "position": 0.0,
                "birth": birth,
                "death": 0.0,
            }
        return {
            "height": remaining / 2,
            "position": remaining / 2,
            "birth": birth,
            "death": death,
        }

    def sample_gamma_prior(self, n=1):
        return self.rng.gamma(self.alpha, 1 / self.beta, n)

    def sample_poisson_prior(self, n=1):
        out = np.zeros(n, dtype=int)
        for i in range(n):
            while True:
                k = self.rng.poisson(self.param)
                if k <= self.kmax:
                    out[i] = k
                    break
        return out

    def step(self, state=None):
        """Advance the stored state by one move."""
        if state is None:
            state = self.state
        self.state, name, accepted = super().step(state)
        return self.state, name, accepted

    def run(self, n_steps):
        """Continue the chain from the stored state."""
        return super().run(self.state, n_steps)


# Add the log likelihood and priors before implementing concrete BaseMove
# subclasses (e.g. birth, death, rate change, and position change). Each
# move will return a Proposal and use its own derived acceptance probability.
