"""The seven moves for the graphical continuous Lyapunov model sampler.

Each move implements its simplified acceptance ratio in the coordinates
(G, mu, r, eta, b_G). The sampler supplies move-selection probabilities.
"""

from dataclasses import replace
from math import log

import numpy as np

from examples.gclm.gclm import (
    GCLMCalculations,
    GCLMPriorConfig,
    GCLMProposalConfig,
)
from examples.gclm.state import State
from rjmcmc.rjmcmc import BaseMove, Proposal


def _accept(log_ratio, rng):
    """Accept with probability min(1, exp(log_ratio))."""
    # -Exp(1) has the same distribution as log(U), U ~ Uniform(0, 1).
    return bool(-rng.exponential() < log_ratio)


class _GCLMMove(BaseMove):
    """Shared data calculations and prior/proposal settings for GCLM moves."""

    def __init__(
        self,
        calculations: GCLMCalculations,
        prior_config: GCLMPriorConfig,
        proposal_config: GCLMProposalConfig,
    ):
        self.calculations = calculations
        self.prior = prior_config
        self.proposal = proposal_config

    def _log_likelihood_ratio(self, state: State, proposed: State) -> float:
        """Return log L(proposed) - log L(state); reject a failed candidate solve."""
        old = self.calculations.log_likelihood(state)
        try:
            new = self.calculations.log_likelihood(proposed, proposed=True)
        except np.linalg.LinAlgError:
            return -np.inf
        return new - old


class MeanMove(_GCLMMove):
    """Draw the entire mean vector from its Gaussian full conditional."""

    def __init__(self, calculations, prior_config, proposal_config):
        super().__init__(calculations, prior_config, proposal_config)
        self._eigen_key = None
        self._eigenvalues = None
        self._eigenvectors = None

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        calculation = self.calculations.drift(state)
        # One bounded eigendecomposition cache: mean moves leave Sigma fixed.
        if calculation.key != self._eigen_key:
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(
                calculation.covariance
            )
            if np.any(self._eigenvalues <= 0):
                raise np.linalg.LinAlgError("Covariance eigenvalues must be positive")
            self._eigen_key = calculation.key

        eigenvalues = self._eigenvalues
        eigenvectors = self._eigenvectors
        prior_variance = self.prior.s_mu**2
        n_prior_variance = self.calculations.n * prior_variance
        denominator = eigenvalues + n_prior_variance
        mean_weights = n_prior_variance / denominator
        posterior_std = self.prior.s_mu * np.sqrt(eigenvalues / denominator)

        centered_data = self.calculations.y_bar - self.prior.m_0
        posterior_mean = (
            self.prior.m_0
            + eigenvectors @ (mean_weights * (eigenvectors.T @ centered_data))
        )
        new_mu = posterior_mean + eigenvectors @ (
            posterior_std * rng.standard_normal(self.calculations.p)
        )
        return Proposal(replace(state, mu=new_mu), "mean")

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        # The proposal is the full conditional, so this is a Gibbs step.
        return True


class ContrastMove(_GCLMMove):
    """Gaussian random walk on all p-1 diagonal contrast coordinates."""

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        new_r = state.r + rng.normal(0.0, self.proposal.sigma_r, size=state.r.size)
        if not np.all(np.isfinite(new_r)):
            return Proposal(state, "contrast")
        return Proposal(replace(state, r=new_r), "contrast", u=True)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        log_ratio -= (
            np.dot(proposal.state.r, proposal.state.r) - np.dot(state.r, state.r)
        ) / (2.0 * self.prior.s_r**2)
        return _accept(log_ratio, rng)


class MarginMove(_GCLMMove):
    """Gaussian random walk on log(eta), preserving a positive margin."""

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        log_eta = log(state.eta)
        with np.errstate(over="ignore", under="ignore"):
            new_eta = np.exp(log_eta + rng.normal(0.0, self.proposal.sigma_eta))
        if not np.isfinite(new_eta) or new_eta <= 0:
            return Proposal(state, "margin")
        return Proposal(replace(state, eta=new_eta), "margin", u=True)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        old_log_eta = log(state.eta) - self.prior.mu_eta
        new_log_eta = log(proposal.state.eta) - self.prior.mu_eta
        # The 1/eta term in the lognormal prior cancels the proposal ratio.
        log_ratio -= (new_log_eta**2 - old_log_eta**2) / (
            2.0 * self.prior.s_eta**2
        )
        return _accept(log_ratio, rng)


class CoefficientMove(_GCLMMove):
    """Update a Bernoulli-selected subset of the existing edge coefficients."""

    def available(self, state):
        return state.graph.k > 0

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        selected = np.flatnonzero(
            rng.random(state.graph.k) < self.proposal.gamma
        )
        if selected.size == 0:
            # The empty subset is a no-op, not an accepted coefficient change.
            return Proposal(state, "coefficient")
        new_b = state.b.copy()
        new_b[selected] += rng.normal(
            0.0, self.proposal.sigma_b, size=selected.size
        )
        if not np.all(np.isfinite(new_b)):
            return Proposal(state, "coefficient")
        return Proposal(replace(state, b=new_b), "coefficient", u=selected)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        selected = proposal.u
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        log_ratio -= (
            np.dot(proposal.state.b[selected], proposal.state.b[selected])
            - np.dot(state.b[selected], state.b[selected])
        ) / (2.0 * self.prior.tau**2)
        return _accept(log_ratio, rng)


class ReversalMove(_GCLMMove):
    """Reverse one eligible edge and draw its new coefficient from the prior."""

    def available(self, state):
        return bool(state.graph.reversible_edges())

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        reversible = state.graph.reversible_edges()
        edge = reversible[int(rng.integers(len(reversible)))]
        coefficient = rng.normal(0.0, self.prior.tau)
        if not np.isfinite(coefficient):
            return Proposal(state, "reversal")
        graph = state.graph.reverse_edge(edge)
        coefficients = state.coefficients_by_edge
        del coefficients[edge]
        coefficients[(edge[1], edge[0])] = coefficient
        proposed = State.from_coefficients(
            graph, state.mu, state.r, state.eta, coefficients
        )
        return Proposal(proposed, "reversal", u=edge)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        # Graph prior, move/edge choice, and coefficient prior/proposal cancel.
        return _accept(self._log_likelihood_ratio(state, proposal.state), rng)


class BirthMove(_GCLMMove):
    """Add one admissible edge and draw its coefficient from the prior."""

    def _birth_edges(self, state):
        return state.graph.birth_edges(simple=self.prior.graph_space == "simple")

    def available(self, state):
        return bool(self._birth_edges(state))

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        birth_edges = self._birth_edges(state)
        edge = birth_edges[int(rng.integers(len(birth_edges)))]
        coefficient = rng.normal(0.0, self.prior.tau)
        if not np.isfinite(coefficient):
            return Proposal(state, "death")
        graph = state.graph.add_edge(edge, simple=self.prior.graph_space == "simple")
        coefficients = state.coefficients_by_edge
        coefficients[edge] = coefficient
        proposed = State.from_coefficients(
            graph, state.mu, state.r, state.eta, coefficients
        )
        return Proposal(proposed, "death", u=edge)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        if proposal.u is None:
            return False
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        log_ratio += log(self.prior.lambda_p) - log(state.graph.k + 1)
        log_ratio += log(reverse_probability) - log(forward_probability)
        return _accept(log_ratio, rng)


class DeathMove(_GCLMMove):
    """Delete one uniformly chosen off-diagonal edge and its coefficient."""

    def available(self, state):
        return state.graph.k > 0

    def propose(self, state: State, rng: np.random.Generator) -> Proposal:
        edges = state.graph.ordered_edges
        edge = edges[int(rng.integers(len(edges)))]
        graph = state.graph.remove_edge(edge)
        coefficients = state.coefficients_by_edge
        del coefficients[edge]
        proposed = State.from_coefficients(
            graph, state.mu, state.r, state.eta, coefficients
        )
        return Proposal(proposed, "birth", u=edge)

    def acceptance(
        self, state, proposal, forward_probability, reverse_probability, rng
    ):
        log_ratio = self._log_likelihood_ratio(state, proposal.state)
        log_ratio += log(state.graph.k) - log(self.prior.lambda_p)
        log_ratio += log(reverse_probability) - log(forward_probability)
        return _accept(log_ratio, rng)
