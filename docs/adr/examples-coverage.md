# Examples coverage

This page tracks which pipeline-stage topics have a worked example or user-guide page, which
are partially covered, and which are still unwritten. The grouping mirrors the structure
fixed by [ADR 0014](https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/main/docs/adr/0014-group-docs-by-pipeline-stage.md);
that ADR records the *decision*, this page is the *living inventory*.

## Status rules

Mark a topic covered (✅) only when all of the following hold:

- The example runs end-to-end under Kaimon.
- `pre-commit run -a` passes.
- The page opens with a `!!! tip "When to reach for this"` admonition.
- The page ends with a digestible visualisation and carries a `#src` `## Findings` block.

| Symbol | Meaning |
| ------ | ------- |
| ✅ | Covered — example or user-guide page exists and runs. |
| ⬜ | Not yet written. |
| 🔶 | Partially covered — appears as a supporting detail inside another page but deserves its own example. |

## 1. Foundations

| Status | Topic | File |
| ------ | ----- | ---- |
| ✅ | Getting started — `prices_to_returns`, `ReturnsResult`, basic `MeanRisk` solve | `examples/1_foundations/01_Getting_Started.jl` |
| ✅ | Data preprocessing and imputation — `missing_col_percent`, `missing_row_percent`, `Impute.jl` imputors (`LOCF`, `Interpolate`), handling stale prices and gaps before `prices_to_returns` | `examples/1_foundations/02_Data_Preprocessing.jl` |

## 2. Moments and priors

| Status | Topic | File |
| ------ | ----- | ---- |
| ✅ | Expected returns estimation (`SimpleExpectedReturns`, `ShrunkExpectedReturns`, `EquilibriumExpectedReturns`, windowed variants) | `examples/2_moments_priors/01_Expected_Returns_Estimation.jl` |
| ✅ | Covariance estimation (classical, Gerber, Smyth-Broby, mutual info, distance, denoising, detoning) | `examples/2_moments_priors/02_Covariance_Estimation.jl` |
| ✅ | Higher-moment estimation (coskewness, cokurtosis, `HighOrderPriorEstimator`) | `examples/2_moments_priors/03_Higher_Moment_Estimation.jl` |
| ✅ | Factor priors (`FactorPrior`, `HighOrderFactorPriorEstimator`, stepwise and dimension-reduction regression) | `examples/2_moments_priors/04_Factor_Priors.jl` |
| ✅ | Black-Litterman prior (views, `views_conf`, equilibrium prior, posterior) | `examples/2_moments_priors/05_Black_Litterman.jl` |
| ✅ | Advanced Black-Litterman variants — `BayesianBlackLittermanPrior`, `FactorBlackLittermanPrior` (`rsd`/`l`), `AugmentedBlackLittermanPrior` (asset + factor views) | `examples/2_moments_priors/06_Advanced_Black_Litterman.jl` |
| ✅ | Entropy pooling | `examples/2_moments_priors/07_Entropy_Pooling.jl` |
| ✅ | Opinion pooling | `examples/2_moments_priors/08_Opinion_Pooling.jl` |
| ✅ | Uncertainty sets (`NormalUncertaintySet` box/ellipsoidal, `q` sweep, mean and covariance forms, `DeltaUncertaintySet`; `ARCHUncertaintySet` block bootstrap described) | `examples/2_moments_priors/09_Uncertainty_Sets.jl` |
| ✅ | Windowed moment estimators — `WindowedCovariance`, `WindowedExpectedReturns` (trailing `window`, index-vector window for a named episode, `eweights` observation weights, prior wiring, caller-driven rolling demo) | `examples/2_moments_priors/10_Windowed_Estimators.jl` |
| ⬜ | Regime-adjusted estimators — `RegimeAdjustedExpWeightedCovariance` and `RegimeAdjustedExpWeightedVariance`; covariance conditioned on a regime model | — |
| 🔶 | Implied volatility estimator — `ImpliedVolatility` (requires live Python/PythonCall environment; a stub showing how to plug into `EmpiricalPrior` would be useful) | — |

## 3. Optimisers

| Status | Topic | File |
| ------ | ----- | ---- |
| ✅ | `MeanRisk` objectives (`MinimumRisk`, `MaximumReturn`, `MaximumRatio`, `MaximumUtility`) | `examples/3_optimisers/01_MeanRisk_Objectives.jl` |
| ✅ | Efficient frontier — both directions, `Frontier` helper, `NearOptimalCentering` frontier | `examples/3_optimisers/02_Efficient_Frontier.jl` |
| ✅ | Pareto surface — multi-objective, `plot_measures` | `examples/3_optimisers/03_Pareto_Surface.jl` |
| ✅ | Multiple risk measures — vector of risk measures, scalariser sweep, `HighOrderPriorEstimator` | `examples/3_optimisers/04_Multiple_Risk_Measures.jl` |
| ✅ | OWA risk measures — `OrderedWeightsArray` with closed-form vectors (`owa_gmd`, `owa_cvar`, `owa_tg`, `owa_tgrg`, `owa_wr`, `owa_rg`, `owa_cvarrg`), approximate formulation, `OrderedWeightsArrayRange`, L-moment CRM with `NormalisedConstantRelativeRiskAversion` | `examples/3_optimisers/05_OWA_Risk_Measures.jl` |
| ✅ | `BrownianDistanceVariance` and `VarianceSkewKurtosis` — NormOneCone and inequality formulations, both rank-expression variants; `VarianceSkewKurtosis` with `HighOrderPriorEstimator` + SCS, custom skewness/kurtosis scales | `examples/3_optimisers/06_Brownian_Distance_Variance_and_VarianceSkewKurtosis.jl` |
| ✅ | Drawdown-family risk measures — `AverageDrawdown`, `UlcerIndex`, `MaximumDrawdown`, `DrawdownatRisk`, `ConditionalDrawdownatRisk`; alpha sweep, drawdown upper-bound constraint, post-optimisation `drawdowns()` / `cumulative_returns()` analytics | `examples/3_optimisers/07_Drawdown_Risk_Measures.jl` |
| ✅ | Exotic / tail risk measures — `EntropicValueatRisk`, `RelativisticValueatRisk`, `PowerNormValueatRisk`, `GenericValueatRiskRange`; cross-evaluation vs CVaR, κ-interpolation (EVaR↔worst realisation), `p`-sweep, asymmetric two-sided range | `examples/3_optimisers/08_Exotic_Tail_Risk_Measures.jl` |
| ✅ | Risk budgeting — `AssetRiskBudgeting`, `RelaxedRiskBudgeting`, `FactorRiskBudgeting` | `examples/3_optimisers/09_Risk_Budgeting.jl` |
| ✅ | Risk contribution — asset and factor variance risk contributions using `rc` constraints, objective-function sweep (`MinimumRisk`, `MaximumUtility`, `MaximumRatio`) | `examples/3_optimisers/10_Risk_Contribution.jl` |
| ✅ | Clustering optimisers — `HierarchicalRiskParity`, `HierarchicalEqualRiskContribution`, `SchurComplementHierarchicalRiskParity` | `examples/3_optimisers/11_Clustering_Optimisers.jl` |
| ✅ | Clustering optimisers with mixed risks — mixed risk measures, scalariser sweep, constrained `HierarchicalEqualRiskContribution` | `examples/3_optimisers/12_Clustering_Mixed_Risks_And_Constraints.jl` |
| ✅ | Meta-optimisers overview — `NestedClustered`, `Stacking`, `SubsetResampling` | `examples/3_optimisers/13_Meta_Optimisers.jl` |
| ✅ | Subset resampling and cross-validation — `SubsetResampling`, `cross_val_predict`, frontier of a meta-optimiser | `examples/3_optimisers/14_Subset_Resampling_and_Cross_Validation.jl` |
| ✅ | Near optimal centering | `examples/3_optimisers/15_Near_Optimal_Centering.jl` |

## 4. Constraints and costs

| Status | Topic | File |
| ------ | ----- | ---- |
| ✅ | Budget constraints | `examples/4_constraints_costs/01_Budget_Constraints.jl` |
| ✅ | Linear and group constraints | `examples/4_constraints_costs/02_Linear_Group_Constraints.jl` |
| ✅ | Cardinality and threshold constraints (`card`, `gcarde`, `scard`/`slt`/`sst`, `sgcarde`/`sglt`/`sgst`; needs a MIP solver) | `examples/4_constraints_costs/03_Cardinality_and_Threshold.jl` |
| ✅ | Phylogeny and centrality constraints | `examples/4_constraints_costs/04_Phylogeny_Centrality.jl` |
| ✅ | Turnover and tracking | `examples/4_constraints_costs/05_Turnover_and_Tracking.jl` |
| ✅ | Fees and net returns | `examples/4_constraints_costs/06_Fees_and_Net_Returns.jl` |
| ✅ | Regularisation and effective assets | `examples/4_constraints_costs/07_Regularisation.jl` |
| ✅ | Nested clustered optimisation with layered constraints and fees | `examples/4_constraints_costs/08_Nested_Clustered_Constraints.jl` |

## 5. Validation and tuning

| Status | Topic | File |
| ------ | ----- | ---- |
| ✅ | Cross-validation — `KFold`, `CombinatorialCrossValidation`, `IndexWalkForward`, `NearestQuantilePrediction` | `examples/5_validation_tuning/01_Cross_Validation.jl` |
| ✅ | Hyperparameter tuning | `examples/5_validation_tuning/02_Hyperparameter_Tuning.jl` |
| ✅ | `DateWalkForward` — calendar-aligned walk-forward using Julia's `Dates` module (month-end `adjuster`, `previous` alignment) | `examples/5_validation_tuning/01_Cross_Validation.jl` (§2.3.2) |

## 6. Post-processing

| Status | Topic | File |
| ------ | ----- | ---- |
| ✅ | Finite allocation — rounding fractional weights to integer lots | `examples/6_post_processing/01_Finite_Allocation.jl` |
| ✅ | Plotting and reporting | `examples/6_post_processing/02_Plotting_and_Reporting.jl` |
| ✅ | Performance attribution — `cumulative_returns`, `drawdowns`, realised-performance scorecard, `calc_net_returns`/`calc_fees` cost drag, `risk_contribution` attribution (post-optimisation diagnostics) | `examples/6_post_processing/03_Performance_Attribution.jl` |

## 7. Putting it together (end-to-end profiles)

| Status | Topic | File |
| ------ | ----- | ---- |
| ✅ | Retail daily profile | `examples/7_putting_it_together/01_Profile_Retail_Daily.jl` |
| ✅ | Desk monthly profile | `examples/7_putting_it_together/02_Profile_Desk_Monthly.jl` |
| ✅ | Institutional profile | `examples/7_putting_it_together/03_Profile_Institutional.jl` |
| ✅ | Factor + Black-Litterman + JuMP pipeline — `FactorBlackLittermanPrior` (factor-premia views) → constrained `MaximumRatio` (per-asset + sector caps) → `DiscreteAllocation`, isolated as a reusable factor-views profile | `examples/7_putting_it_together/04_Profile_Factor_Views.jl` |

## User guide gaps

| Status | Topic | Where |
| ------ | ----- | ----- |
| ✅ | Data preprocessing and imputation before `prices_to_returns` | `user_guide/01_Data_and_Priors.jl` §1 |
| ✅ | OWA vs moment-based vs quantile-based risk measure families — when to choose each | `user_guide/02_Optimisers.jl` §2 |
| ✅ | Drawdown as a risk measure vs as a post-optimisation diagnostic | `user_guide/02_Optimisers.jl` §2 |
| ✅ | The full Black-Litterman family and how the Bayesian, Factor, and Augmented variants differ from the base form | `user_guide/01_Data_and_Priors.jl` §3 |
| ✅ | Windowed and regime-adjusted moment estimators — motivation and trade-offs vs full-sample estimators | `user_guide/01_Data_and_Priors.jl` §3 |
