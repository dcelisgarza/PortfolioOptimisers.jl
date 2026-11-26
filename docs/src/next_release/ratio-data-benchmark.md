```@meta
Description = "The walks of a data benchmark, of the previous weights, of the readers and of the asset view enter the sides of a ratio measure."
```

# [The walks enter the sides of a ratio measure](@id migration-next-ratio-data-benchmark)

## [Calls that now succeed](@id migration-next-ratio-data-benchmark-success)

- **A `DataBenchmarkTracking` inside a `RiskRatio`, a `NonOptimisationRiskRatio`, a `MeanReturnRiskRatio` or an `ExpectedReturnRiskRatio`** reads the benchmark of the returns data. Before this release, `HierarchicalRiskParity`, `HierarchicalEqualRiskContribution`, `RiskLoss` and `expected_risk` raised an `ArgumentError`, because no reader bound it.
- **A ratio measure or an `ExpectedReturn` that holds per-asset data, with an asset that is not investable.** The data are a covariance matrix, benchmark weights, turnover weights or expected returns. Before this release, the call raised a `DimensionMismatch`, because it did not cut the parts of the measure to the investable assets.

## [Results that move](@id migration-next-ratio-data-benchmark-numbers)

- **The walk-forward of an optimiser whose `RiskRatio` holds previous weights**, such as a `TurnoverRiskMeasure` or a `WeightsTracking` whose `fixed` is `false`, runs its folds in sequence and gives each fold the weights of the fold before. Before this release, the folds ran in parallel, and each fold read the weights that the caller gave.

## [Warnings that are new](@id migration-next-ratio-data-benchmark-warnings)

- **A `RiskRatio` or a `NonOptimisationRiskRatio` that holds a reader with no `x_src`**, such as an `UncertaintySetVariance` whose uncertainty set has a prior of its own, in a head with a reader that states `x_src = ExcessSource()`. The head names the reader in its warning, as it does for the other measures. Such a reader that states `x_src = ExcessSource()` also drops the observations at which a vector benchmark has a gap.

## [If you extend the library](@id migration-next-ratio-data-benchmark-extending)

- **A risk measure of your own that holds a tracking part** and adds a method of the internal `PortfolioOptimisers.bind_data_benchmark` also adds one of `PortfolioOptimisers.holds_data_benchmark`, which the bind of a vector and of a composite reads first. A measure that declares the part in `PortfolioOptimisers.deferred_slots` needs neither.
