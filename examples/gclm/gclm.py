"""GCLM RJMCMC sampler and its shared numerical calculations.

The sampler and its moves will use one GCLMCalculations instance. Its two
cache slots hold only the current and proposed drift calculations, not the
full history of states.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvals, solve_continuous_lyapunov

from examples.gclm.state import Graph, State
from rjmcmc.rjmcmc import BaseRJMCMC


@dataclass(frozen=True, eq=False)
class DriftCalculation:
    """Expensive quantities shared by states with the same drift parameters."""

    key: tuple
    H: np.ndarray
    spectral_abscissa: float
    M: np.ndarray
    covariance: np.ndarray
    cholesky: tuple[np.ndarray, bool]
    log_determinant: float
    empirical_trace: float
    relative_lyapunov_residual: float


class GCLMCalculations:
    """Precompute data summaries and evaluate a GCLM state with C = 2 I_p.

    The log-likelihood omits the state-independent term -np log(2*pi)/2.
    Use proposed=True for a candidate drift, then call finish_proposal with
    the returned state to promote or discard its cached calculation.
    """

    def __init__(self, observations, residual_tolerance=1e-8):
        y = np.asarray(observations, dtype=float)
        if y.ndim != 2 or y.shape[0] < 1 or y.shape[1] < 2:
            raise ValueError("observations must have shape (n, p) with n >= 1 and p >= 2")
        if not np.all(np.isfinite(y)):
            raise ValueError("observations must be finite")
        if not np.isfinite(residual_tolerance) or residual_tolerance <= 0:
            raise ValueError("residual_tolerance must be finite and positive")

        self.n, self.p = y.shape
        self.y_bar = y.mean(axis=0)
        centered = y - self.y_bar
        self.empirical_covariance = centered.T @ centered / self.n
        if not np.all(np.isfinite(self.y_bar)) or not np.all(
            np.isfinite(self.empirical_covariance)
        ):
            raise ValueError("Data summaries must be finite")
        self.contrast_basis = self._contrast_basis(self.p)
        self.residual_tolerance = float(residual_tolerance)
        self._volatility = 2.0 * np.eye(self.p)
        self._current: DriftCalculation | None = None
        self._proposed: DriftCalculation | None = None

        for array in (self.y_bar, self.empirical_covariance, self.contrast_basis):
            array.setflags(write=False)

    @staticmethod
    def _contrast_basis(p):
        """Orthonormalize [1, e_1, ..., e_{p-1}] and omit the first column."""
        vectors = np.column_stack((np.ones(p), np.eye(p)[:, :p - 1]))
        orthogonal, _ = np.linalg.qr(vectors)
        return orthogonal[:, 1:]

    @staticmethod
    def _drift_key(state):
        # A mean update leaves this key unchanged. State arrays are float64.
        # The first three items also identify H, which is unchanged by eta.
        return (state.graph, state.r.tobytes(), state.b.tobytes(), state.eta)

    def drift(self, state: State, *, proposed=False) -> DriftCalculation:
        """Return H, phi(H), M, Sigma, and reusable likelihood quantities."""
        if not isinstance(state, State):
            raise TypeError("state must be a GCLM State")
        if state.graph.p != self.p:
            raise ValueError("state graph dimension does not match the data")
        key = self._drift_key(state)
        for cached in (self._current, self._proposed):
            if cached is not None and cached.key == key:
                return cached

        result = self._calculate_drift(state, key)
        if proposed:
            self._proposed = result
        else:
            self._current = result
        return result

    def _calculate_drift(self, state, key):
        # A margin move changes eta only, so reuse H and its eigenvalue result.
        shared_H = next(
            (
                cached for cached in (self._current, self._proposed)
                if cached is not None and cached.key[:3] == key[:3]
            ),
            None,
        )
        if shared_H is not None:
            H = shared_H.H
            spectral_abscissa = shared_H.spectral_abscissa
        else:
            H = np.diag(self.contrast_basis @ state.r)
            for (i, j), coefficient in zip(state.graph.ordered_edges, state.b):
                H[j, i] = coefficient
            eigenvalues = eigvals(H)
            if not np.all(np.isfinite(eigenvalues)):
                raise np.linalg.LinAlgError("Could not compute finite eigenvalues of H")
            spectral_abscissa = float(np.max(eigenvalues.real))
        M = H - (spectral_abscissa + state.eta) * np.eye(self.p)
        if not np.all(np.isfinite(M)):
            raise np.linalg.LinAlgError("Constructed drift matrix is not finite")

        # scipy solves M Sigma + Sigma M.T = Q; the model has Q = -2 I_p.
        covariance = solve_continuous_lyapunov(M, -self._volatility)
        if not np.all(np.isfinite(covariance)):
            raise np.linalg.LinAlgError("Lyapunov solution is not finite")
        covariance = (covariance + covariance.T) / 2.0

        left = M @ covariance
        right = covariance @ M.T
        residual = left + right + self._volatility
        denominator = (
            np.linalg.norm(left, ord="fro")
            + np.linalg.norm(right, ord="fro")
            + np.linalg.norm(self._volatility, ord="fro")
        )
        relative_residual = float(np.linalg.norm(residual, ord="fro") / denominator)
        if not np.isfinite(relative_residual) or relative_residual > self.residual_tolerance:
            raise np.linalg.LinAlgError("Lyapunov residual exceeds tolerance")

        # A successful Cholesky factorization checks positive definiteness.
        # Compute log_determinant via the Cholesky factor.
        cholesky = cho_factor(covariance, lower=True, check_finite=True)
        log_determinant = float(2.0 * np.log(np.diag(cholesky[0])).sum())
        empirical_trace = float(np.trace(
            cho_solve(cholesky, self.empirical_covariance, check_finite=True)
        ))
        if not np.isfinite(log_determinant) or not np.isfinite(empirical_trace):
            raise np.linalg.LinAlgError("Likelihood quantities are not finite")

        for array in (H, M, covariance, cholesky[0]):
            array.setflags(write=False)
        return DriftCalculation(
            key=key,
            H=H,
            spectral_abscissa=spectral_abscissa,
            M=M,
            covariance=covariance,
            cholesky=cholesky,
            log_determinant=log_determinant,
            empirical_trace=empirical_trace,
            relative_lyapunov_residual=relative_residual,
        )

    def log_likelihood(self, state: State, *, proposed=False) -> float:
        """Evaluate the Gaussian log-likelihood up to a constant."""
        calculation = self.drift(state, proposed=proposed)
        difference = self.y_bar - state.mu
        quadratic = float(difference @ cho_solve(calculation.cholesky, difference))
        value = -0.5 * self.n * (
            calculation.log_determinant
            + calculation.empirical_trace
            + quadratic
        )
        if not np.isfinite(value):
            raise np.linalg.LinAlgError("Log-likelihood is not finite")
        return float(value)

    def finish_proposal(self, accepted: bool, returned_state: State) -> None:
        """Keep the current cache aligned with the state returned by a move."""
        if accepted and self._proposed is not None:
            self._current = self._proposed
        self._proposed = None
        # Also handles moves that did not precompute their proposed drift.
        self.drift(returned_state)


_MOVE_NAMES = (
    "mean", "contrast", "margin", "coefficient", "reversal", "birth", "death",
)


@dataclass(frozen=True)
class GCLMPriorConfig:
    """Graph space and hyperparameters defining the statistical model.

    s_r, s_eta, tau, and s_mu are standard deviations, not variances.
    """

    graph_space: str
    lambda_p: float
    s_r: float
    mu_eta: float
    s_eta: float
    tau: float
    m_0: float
    s_mu: float

    def __post_init__(self):
        if self.graph_space not in ("general", "simple"):
            raise ValueError("graph_space must be 'general' or 'simple'")
        for name in ("lambda_p", "s_r", "s_eta", "tau", "s_mu"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        for name in ("mu_eta", "m_0"):
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True, eq=False)
class GCLMProposalConfig:
    """Proposal scales, coefficient-selection probability, and move weights."""

    sigma_r: float
    sigma_eta: float
    sigma_b: float
    gamma: float
    move_weights: Mapping[str, float] | None = None

    def __post_init__(self):
        for name in ("sigma_r", "sigma_eta", "sigma_b"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if not np.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        object.__setattr__(self, "gamma", float(self.gamma))

        weights = (
            {name: 1.0 for name in _MOVE_NAMES}
            if self.move_weights is None else self.move_weights
        )
        if not isinstance(weights, Mapping) or set(weights) != set(_MOVE_NAMES):
            raise ValueError("Give a base weight for every GCLM move")
        if any(not np.isfinite(weight) or weight <= 0 for weight in weights.values()):
            raise ValueError("Base move weights must be finite and positive")
        # Copy before freezing so changes to the input dictionary cannot leak in.
        frozen_weights = MappingProxyType({
            name: float(weights[name]) for name in _MOVE_NAMES
        })
        object.__setattr__(self, "move_weights", frozen_weights)


class GCLM(BaseRJMCMC):
    """RJMCMC sampler for general or simple graphical Lyapunov models.

    Pass move objects that share this sampler's calculations and config
    instances. The moves implement their proposals and acceptance rules;
    BaseRJMCMC selects moves using the graph-dependent weights below.
    Prior and proposal settings are stored separately in frozen configs.
    """

    MOVE_NAMES = _MOVE_NAMES

    def __init__(
        self,
        moves,
        calculations: GCLMCalculations,
        prior_config: GCLMPriorConfig,
        proposal_config: GCLMProposalConfig,
        *,
        initial_state: State | None = None,
        seed=None,
    ):
        if not isinstance(calculations, GCLMCalculations):
            raise TypeError("calculations must be a GCLMCalculations instance")
        if not isinstance(prior_config, GCLMPriorConfig):
            raise TypeError("prior_config must be a GCLMPriorConfig")
        if not isinstance(proposal_config, GCLMProposalConfig):
            raise TypeError("proposal_config must be a GCLMProposalConfig")
        if set(moves) != set(self.MOVE_NAMES):
            raise ValueError("Provide every GCLM move exactly once")
        for name, move in moves.items():
            if getattr(move, "calculations", None) is not calculations:
                raise ValueError(f"{name} move must use the shared calculations")
            if getattr(move, "prior", None) is not prior_config:
                raise ValueError(f"{name} move must use the shared prior_config")
            if getattr(move, "proposal", None) is not proposal_config:
                raise ValueError(f"{name} move must use the shared proposal_config")
        self.prior_config = prior_config
        self.proposal_config = proposal_config
        self.calculations = calculations
        super().__init__(moves, self._gclm_move_weights, seed)

        if initial_state is None:
            # Start in the empty graph at the prior centers.
            initial_state = State(
                graph=Graph(calculations.p, frozenset()),
                mu=np.full(calculations.p, prior_config.m_0),
                r=np.zeros(calculations.p - 1),
                eta=np.exp(prior_config.mu_eta),
                b=np.empty(0),
            )
        if not isinstance(initial_state, State):
            raise TypeError("initial_state must be a GCLM State")
        initial_state.validate_for_model(calculations.p, prior_config.graph_space)
        self.state = initial_state
        self.calculations.finish_proposal(False, initial_state)

    def _gclm_move_weights(self, state: State) -> dict[str, float]:
        """Graph-dependent unnormalized weights from the thesis."""
        graph = state.graph
        max_edges = (
            graph.p * (graph.p - 1)
            if self.prior_config.graph_space == "general"
            else graph.p * (graph.p - 1) // 2
        )
        rho = graph.k / max_edges
        weights = self.proposal_config.move_weights
        return {
            "mean": weights["mean"],
            "contrast": weights["contrast"],
            "margin": weights["margin"],
            "coefficient": rho * weights["coefficient"],
            "reversal": (
                rho * weights["reversal"]
                if graph.reversible_edges() else 0.0
            ),
            # A birth edge exists exactly when rho < 1 in either graph space.
            "birth": (1.0 - rho) * weights["birth"],
            "death": rho * weights["death"],
        }

    def step(self, state=None):
        """Advance the stored state and keep the drift cache in sync."""
        if state is None:
            state = self.state
        self.calculations.drift(state)
        try:
            next_state, name, accepted = super().step(state)
        except Exception:
            self.calculations.finish_proposal(False, state)
            raise
        self.calculations.finish_proposal(accepted, next_state)
        self.state = next_state
        return next_state, name, accepted

    def run(self, n_steps):
        """Continue the chain from its stored state."""
        return super().run(self.state, n_steps)
