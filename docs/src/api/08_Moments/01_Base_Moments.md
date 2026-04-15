# Base moments

## Abstract moment types and fallbacks

Some optimisations and constraints make use of summary statistics. These types and functions form the base for moment estimation in `PortfolioOptimisers.jl`.

They also provide generic fallbacks for the various functionality in the library.

```@docs
AbstractExpectedReturnsEstimator
AbstractExpectedReturnsAlgorithm
AbstractMomentAlgorithm
AbstractCovarianceEstimator
AbstractVarianceEstimator
factory(ce::StatsBase.CovarianceEstimator, args...)
robust_cov
robust_cor
moment_window_and_weights
```

## Full and semi moments

Moments other than the expected return can be estimated using the entire spectrum of deviations (full), or only the deviations below a target (semi/downside). These types allow us to provide such functionality.

```@docs
Full
Semi
```
