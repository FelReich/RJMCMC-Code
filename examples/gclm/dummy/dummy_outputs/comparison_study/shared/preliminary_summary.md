# Preliminary GCLM comparison

Each experiment has five independent chains, 100,000 updates and 50,000 burn-in per chain. A and B use exactly the same generated observations.

| Experiment | Mean k | P(true graph) | Covariance error | Drift error | Max M R-hat | Min M ESS/chain | Max Sigma R-hat | Min Sigma ESS/chain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_simple_on_simple | 2.612 | 0.1542 | 0.475 | 0.565 | 1.001 | 507 | 1.000 | 3608 |
| B_general_on_simple | 3.568 | 0.0817 | 0.478 | 0.561 | 1.004 | 207 | 1.000 | 2742 |
| C_general_on_nonsimple | 2.278 | 0.0089 | 0.368 | 0.989 | 1.003 | 333 | 1.000 | 3358 |

For the shared simple-graph data, A and B give nearly the same covariance errors (0.475 and 0.478) and mean marginal 95% covariance interval widths (0.461 and 0.459). The mean marginal 95% interval width across off-diagonal drift entries is 1.131 under simple inference versus 2.436 under general inference. The simple restriction narrows the drift marginals in this dataset, but its worst drift-entry ESS is still very low.

The generating two-edge graph has prior probability 0.0288 in A and 0.0155 in B. The corresponding posterior probabilities are 0.1542 and 0.0817. Thus the graph-space comparison also changes the prior probability of the generating graph.

For non-simple data in C, the posterior covariance mean has Frobenius error 0.368 (empirical covariance: 0.385). The generating graph has posterior probability 0.0089. The largest drift-entry R-hat is 1.003, versus 1.000 for covariance; drift and skew-direction cumulative means remain more variable. This supports a computational mixing concern for drift, but the generating drift need not be recovered from covariance-only observations.

Across all updates, the death-move acceptance rates are 5.3% (A), 16.0% (B), and 33.1% (C); reversal rates are 7.0%, 7.3%, and 15.3%. These are possible proposal bottlenecks to study before the final simulation study.

The covariance error is the Frobenius distance from the generating stationary covariance. The drift error uses the generating drift; it need not be small where drift is non-identifiable.

Convergence flags in each diagnostics.csv use R-hat > 1.05 or a minimum approximate single-chain ESS < 100. Small R-hat alone does not establish convergence. The ESS calculation uses FFT autocorrelation with initial positive pairs; it is an approximation.

The A/B contrast combines a change in identifiability with a change in the admissible models and truncated graph prior. It does not isolate identifiability causally.

This is one simulated dataset per generating model. The low drift and graph ESS values, and the drift R-hat above 1.05 in C, need attention before using these runs as a final thesis simulation study. No automatic tuning or new move types were used.

Five useful meeting figures:

1. `shared/A_vs_B_posterior_k.pdf`
2. `shared/A_vs_B_edge_inclusion.pdf`
3. `shared/A_vs_B_covariance.pdf`
4. `shared/A_vs_B_drift.pdf`
5. `C_general_on_nonsimple/skew_cumulative_means.pdf`
