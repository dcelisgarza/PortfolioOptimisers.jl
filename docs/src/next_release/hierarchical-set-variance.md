```@meta
Description = "The hierarchical optimisers and RiskLoss read the worst case of an UncertaintySetVariance, over one set fitted on the data of the optimisation."
```

# [The fitted set of an uncertainty set variance](@id migration-next-hierarchical-set-variance)

## [Results that move](@id migration-next-hierarchical-set-variance-numbers)

Each entry is about an `UncertaintySetVariance` with an estimator in `ucs`, such as the default `NormalUncertaintySet()`. A measure with a fitted set gives the numbers of the previous release. A `JuMP` optimiser now fits the set once, and every part of the optimisation reads it. In the previous release only the rows of the model read the set, and the other parts read the nominal variance.

- **A `Frontier` in `settings.ub`** spans the worst-case variance of the two corner portfolios. The previous release took the levels from the nominal variance, so the first point of the sweep was infeasible.
- **`RiskTrackingRiskMeasure` and `RiskTrackingError` with `DependentVariableTracking`** compare the worst-case variance of the portfolio with the worst-case variance of the benchmark. The previous release compared it with the nominal variance of the benchmark.
- **The field `r` of the result of a `JuMP` optimiser** holds the fitted set. So `expected_risk(res.r, res.w, res.pr)` gives the worst case that the optimiser read, where the previous release gave the nominal variance.
- **`NearOptimalCentering` with an estimator that reads the prior result**, such as `NormalUncertaintySet(; pe = nothing)` or `OrthogonalUncertaintySet()`, computes the risks of its anchors on the worst case. The previous release computed them on the nominal variance, and fitted the set again for each anchor.
- **`HierarchicalRiskParity` and `HierarchicalEqualRiskContribution`** fit the set once on the returns data and the prior of the optimisation, and allocate on the worst case. The previous release allocated on the nominal variance, so the default `UncertaintySetVariance()` gave the weights of `Variance()`. The same holds for an `UncertaintySetVariance` inside a `RiskRatio`, a `RiskTrackingRiskMeasure` or a vector of measures. An estimator with its own `pe` reads the returns data, so a call with a prior result in `opt.pe` and no returns data raises an `IsNothingError`. Pass the returns data, or set `pe = nothing` in the estimator, which then reads the prior result. An estimator that draws samples gives other weights at each call, unless it states a seed.
- **`RiskLoss`** takes the gradient of the worst case, for an `UncertaintySetVariance` and for an `ArithmeticReturn` with an estimator in `ucs` inside an `ExpectedReturn` or an `ExpectedReturnRiskRatio`. The previous release took the gradient of the nominal value.

## [Calls that now succeed](@id migration-next-hierarchical-set-variance-success)

- **`NearOptimalCentering` with a risk measure that holds per-asset data, or an `UncertaintySetVariance` with an estimator, and an asset that is not investable.** The previous release raised a `DimensionMismatch`, because it did not cut the risk measure to the investable assets.
- **A `ProgrammeAllocationSet` whose ceiling `r` or tracking error `tr` holds an `UncertaintySetVariance` with an estimator that has its own `pe`.** The previous release raised an `IsNothingError`, because the fit got no rows.
- **A `RiskTrackingError` whose measure holds an estimator in a slot**, such as `Variance(; sigma = PortfolioOptimisersCovariance())`, raised a `MethodError` in the previous release.

## [If you extend the library](@id migration-next-hierarchical-set-variance-extending)

- **`PortfolioOptimisers.ucs_risk_measure(r, rd)`**, an internal function, takes the prior result too: `ucs_risk_measure(r, rd, pr)`.
- **`EllipsoidalUncertaintySet` has a fifth field, `chol`**, the optional factor of its shape matrix. The keyword constructor and the positional constructors of three and four arguments do not change. `factory` and the `JuMP` optimisers of an `UncertaintySetVariance` fill it under the `mtx_sqrt` of the measure, so `factory(r, pr).ucs` is a new set and not the set that `r` holds. The value and the model read the factor, and a factor that you state wins over `mtx_sqrt`. A matrix with no square root under `mtx_sqrt` now raises its `LinearAlgebra.PosDefException` in `factory` or in the fit of the set, where the previous release raised it at the first value or at the build of the model.
