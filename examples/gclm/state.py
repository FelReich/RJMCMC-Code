"""Graph and state representation for the GCLM sampler.

Only off-diagonal edges are stored: every vertex has its required self-loop.
An edge (i, j) denotes i -> j, corresponding to drift entry M[j, i].
Vertices use zero-based indices. Coefficients in State.b follow sorted edges.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Graph:
    """A directed graph on p vertices, with implicit diagonal self-loops."""

    p: int
    edges: frozenset[tuple[int, int]]

    def __post_init__(self):
        if isinstance(self.p, (bool, np.bool_)) or not isinstance(
            self.p, (int, np.integer)
        ) or self.p < 1:
            raise ValueError("p must be a positive integer")

        edges = tuple(self.edges)
        for edge in edges:
            if (
                not isinstance(edge, tuple)
                or len(edge) != 2
                or any(
                    isinstance(vertex, (bool, np.bool_))
                    or not isinstance(vertex, (int, np.integer))
                    for vertex in edge
                )
            ):
                raise ValueError("Each edge must be a pair of integer vertices")
            i, j = edge
            if not (0 <= i < self.p and 0 <= j < self.p) or i == j:
                raise ValueError("Edges must join distinct vertices in range(p)")
        if len(set(edges)) != len(edges):
            raise ValueError("Duplicate edges are not allowed")
        object.__setattr__(self, "edges", frozenset(edges))

    @property
    def k(self) -> int:
        """Number of off-diagonal edges."""
        return len(self.edges)

    @property
    def ordered_edges(self) -> tuple[tuple[int, int], ...]:
        """Canonical order used by the coefficient vector b."""
        return tuple(sorted(self.edges))

    @property
    def is_simple(self) -> bool:
        """Whether the graph contains no directed two-cycle."""
        return all((j, i) not in self.edges for i, j in self.edges)

    def birth_edges(self, simple: bool = False) -> tuple[tuple[int, int], ...]:
        """Edges that may be added in the chosen graph space."""
        return tuple(
            (i, j)
            for i in range(self.p)
            for j in range(self.p)
            if i != j
            and (i, j) not in self.edges
            and (not simple or (j, i) not in self.edges)
        )

    def reversible_edges(self) -> tuple[tuple[int, int], ...]:
        """Edges whose reverse is currently absent."""
        return tuple(
            (i, j) for i, j in self.ordered_edges if (j, i) not in self.edges
        )

    def add_edge(self, edge: tuple[int, int], simple: bool = False) -> "Graph":
        """Return a new graph with an admissible edge added."""
        if edge not in self.birth_edges(simple=simple):
            raise ValueError("Edge cannot be added in this graph space")
        return Graph(self.p, self.edges | {edge})

    def remove_edge(self, edge: tuple[int, int]) -> "Graph":
        """Return a new graph without an existing edge."""
        if edge not in self.edges:
            raise ValueError("Edge is not present")
        return Graph(self.p, self.edges - {edge})

    def reverse_edge(self, edge: tuple[int, int]) -> "Graph":
        """Return a new graph with an admissible edge reversed."""
        if edge not in self.reversible_edges():
            raise ValueError("Edge cannot be reversed")
        i, j = edge
        return Graph(self.p, (self.edges - {edge}) | {(j, i)})


@dataclass(frozen=True, eq=False)
class State:
    """One GCLM state (G, mu, r, eta, b_G) in the thesis coordinates.

    b_G follows graph.ordered_edges. The graph, rather than its edge count,
    identifies the statistical model.
    """

    graph: Graph
    mu: np.ndarray
    r: np.ndarray
    eta: float
    b: np.ndarray

    def __post_init__(self):
        if not isinstance(self.graph, Graph):
            raise TypeError("graph must be a Graph")
        for name, length in (
            ("mu", self.graph.p),
            ("r", self.graph.p - 1),
            ("b", self.graph.k),
        ):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.ndim != 1 or len(values) != length:
                raise ValueError(f"{name} must be a vector of length {length}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            values.setflags(write=False)
            object.__setattr__(self, name, values)

        eta = float(self.eta)
        if not np.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        object.__setattr__(self, "eta", eta)
