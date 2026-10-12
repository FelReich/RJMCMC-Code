"""Moves for the piecewise-constant Poisson change-point model.

Register these moves as "height", "position", "birth", and "death". Each
acceptance calculation uses the simplified formula for that specific move.
"""

from math import lgamma, log

import numpy as np

from rjmcmc.rjmcmc import BaseMove, Proposal
from examples.multiple_change_point_model.multiple_change_point_model import (
    State,
    load_disaster_days,
)


def _count_events(days, left, right):
    """Number of events in the half-open interval [left, right)."""
    return np.searchsorted(days, right) - np.searchsorted(days, left)


def _interval_log_likelihood(days, left, right, intensity):
    count = _count_events(days, left, right)
    return count * log(intensity) - (right - left) * intensity


def _accept(log_ratio, rng):
    """Make the accept/reject draw using a log acceptance ratio."""
    # -Exp(1) has the same distribution as log(U) for U ~ Uniform(0, 1).
    return bool(-rng.exponential() < log_ratio)


class _ChangePointMove(BaseMove):
    """Fixed data and prior parameters shared by the four moves."""

    def __init__(self, L, alpha, beta, param, kmax, days=None):
        """Use Gamma(shape=alpha, rate=beta) and truncated Poisson(param)."""
        self.L = L
        self.alpha = alpha
        self.beta = beta
        self.param = param
        self.kmax = kmax
        self.days = np.asarray(
            load_disaster_days() if days is None else days, dtype=float
        ).copy()
        if not np.isfinite(L) or L <= 0:
            raise ValueError("L must be positive")
        if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(beta) or beta <= 0:
            raise ValueError("Gamma prior parameters must be positive")
        if not np.isfinite(param) or param <= 0:
            raise ValueError("Poisson prior parameter must be positive")
        if not isinstance(kmax, (int, np.integer)) or kmax < 0:
            raise ValueError("kmax must be a non-negative integer")
        if self.days.ndim != 1 or np.any(~np.isfinite(self.days)):
            raise ValueError("Event days must be a one-dimensional finite array")
        if (
            np.any(self.days < 0)
            or np.any(self.days >= L)
            or np.any(np.diff(self.days) < 0)
        ):
            raise ValueError("Event days must be sorted and lie in [0, L)")


class HeightMove(_ChangePointMove):
    """Change one intensity by a log-uniform multiplicative factor."""

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        j = int(rng.integers(state.k + 1))
        intensities = state.intensities.copy()
        intensities[j] *= np.exp(rng.uniform(-0.5, 0.5))
        if not np.isfinite(intensities[j]) or intensities[j] <= 0:
            return Proposal(state, "height")
        proposed = State(state.k, state.change_points, intensities)
        return Proposal(proposed, "height", u=j)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        j = proposal.u
        bounds = np.concatenate(([0.0], state.change_points, [self.L]))
        left, right = bounds[j:j + 2]
        old = state.intensities[j]
        new = proposal.state.intensities[j]
        count = _count_events(self.days, left, right)
        log_ratio = (count + self.alpha) * log(new / old)
        log_ratio -= (right - left + self.beta) * (new - old)
        return _accept(log_ratio, rng)


class PositionMove(_ChangePointMove):
    """Move one change point uniformly between its two neighbours."""

    def available(self, state):
        return state.k > 0

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        j = int(rng.integers(state.k))
        bounds = np.concatenate(([0.0], state.change_points, [self.L]))
        left, right = bounds[j], bounds[j + 2]
        new_position = rng.uniform(left, right)
        if not left < new_position < right:
            return Proposal(state, "position")
        change_points = state.change_points.copy()
        change_points[j] = new_position
        proposed = State(state.k, change_points, state.intensities)
        return Proposal(proposed, "position", u=j)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        j = proposal.u
        bounds = np.concatenate(([0.0], state.change_points, [self.L]))
        left, old, right = bounds[j:j + 3]
        new = proposal.state.change_points[j]
        eta_left, eta_right = state.intensities[j:j + 2]

        old_log_likelihood = (
            _interval_log_likelihood(self.days, left, old, eta_left)
            + _interval_log_likelihood(self.days, old, right, eta_right)
        )
        new_log_likelihood = (
            _interval_log_likelihood(self.days, left, new, eta_left)
            + _interval_log_likelihood(self.days, new, right, eta_right)
        )
        log_prior_ratio = (
            log(new - left) + log(right - new)
            - log(old - left) - log(right - old)
        )
        return _accept(new_log_likelihood - old_log_likelihood + log_prior_ratio, rng)


def _birth_log_target_ratio(lower, higher, j, days, L, alpha, beta, param):
    """Log posterior ratio higher/lower for splitting interval j."""
    if param == 0:
        return -np.inf

    bounds = np.concatenate(([0.0], lower.change_points, [L]))
    left, right = bounds[j:j + 2]
    middle = higher.change_points[j]
    old = lower.intensities[j]
    new_left, new_right = higher.intensities[j:j + 2]

    log_likelihood_ratio = (
        _interval_log_likelihood(days, left, middle, new_left)
        + _interval_log_likelihood(days, middle, right, new_right)
        - _interval_log_likelihood(days, left, right, old)
    )
    k = lower.k
    log_k_prior_ratio = log(param) - log(k + 1)
    log_position_prior_ratio = (
        log(2 * k + 3) + log(2 * k + 2) - 2 * log(L)
        + log(middle - left) + log(right - middle) - log(right - left)
    )
    log_intensity_prior_ratio = (
        alpha * log(beta) - lgamma(alpha)
        + (alpha - 1) * (log(new_left) + log(new_right) - log(old))
        - beta * (new_left + new_right - old)
    )
    return (
        log_likelihood_ratio + log_k_prior_ratio
        + log_position_prior_ratio + log_intensity_prior_ratio
    )


def _birth_log_jacobian(lower, higher, j):
    """Log of (eta_left + eta_right)^2 / eta_old."""
    new_left, new_right = higher.intensities[j:j + 2]
    return 2 * np.logaddexp(log(new_left), log(new_right)) - log(lower.intensities[j])


class BirthMove(_ChangePointMove):
    """Insert a uniform change point and split its interval's intensity."""

    def available(self, state):
        return state.k < self.kmax

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        middle = rng.uniform(0, self.L)
        j = int(np.searchsorted(state.change_points, middle))
        bounds = np.concatenate(([0.0], state.change_points, [self.L]))
        left, right = bounds[j:j + 2]
        if not left < middle < right:
            return Proposal(state, "death")

        z = rng.random()
        if z == 0:
            return Proposal(state, "death")
        left_fraction = (middle - left) / (right - left)
        right_fraction = (right - middle) / (right - left)
        log_odds = log(z) - np.log1p(-z)
        log_old = log(state.intensities[j])
        new_left = np.exp(log_old + right_fraction * log_odds)
        new_right = np.exp(log_old - left_fraction * log_odds)
        if (
            not np.isfinite(new_left)
            or not np.isfinite(new_right)
            or min(new_left, new_right) <= 0
        ):
            return Proposal(state, "death")

        change_points = np.insert(state.change_points, j, middle)
        intensities = np.concatenate((
            state.intensities[:j], [new_left, new_right], state.intensities[j + 1:]
        ))
        proposed = State(state.k + 1, change_points, intensities)
        return Proposal(proposed, "death", u=(j, middle, z))

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        j = proposal.u[0]
        log_ratio = _birth_log_target_ratio(
            state, proposal.state, j, self.days, self.L,
            self.alpha, self.beta, self.param,
        )
        log_ratio += log(reverse_probability) - log(forward_probability)
        log_ratio += log(self.L) - log(state.k + 1)
        log_ratio += _birth_log_jacobian(state, proposal.state, j)
        return _accept(log_ratio, rng)


class DeathMove(_ChangePointMove):
    """Remove a uniform change point and merge its adjacent intensities."""

    def available(self, state):
        return state.k > 0

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        j = int(rng.integers(state.k))
        bounds = np.concatenate(([0.0], state.change_points, [self.L]))
        left, middle, right = bounds[j:j + 3]
        eta_left, eta_right = state.intensities[j:j + 2]
        merged = np.exp(
            ((middle - left) * log(eta_left) + (right - middle) * log(eta_right)) 
            / (right - left)
        )
        change_points = np.delete(state.change_points, j)
        intensities = np.concatenate((
            state.intensities[:j], [merged], state.intensities[j + 2:]
        ))
        proposed = State(state.k - 1, change_points, intensities)
        return Proposal(proposed, "birth", u=j)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        j = proposal.u
        lower = proposal.state
        log_ratio = -_birth_log_target_ratio(
            lower, state, j, self.days, self.L,
            self.alpha, self.beta, self.param,
        )
        log_ratio += log(reverse_probability) - log(forward_probability)
        log_ratio += log(state.k) - log(self.L)
        log_ratio -= _birth_log_jacobian(lower, state, j)
        return _accept(log_ratio, rng)
