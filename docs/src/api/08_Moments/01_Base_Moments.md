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
port_opt_view(ce::StatsBase.CovarianceEstimator, ::Any, args...)
factory(ce::StatsBase.CovarianceEstimator, args...)
port_opt_view(me::AbstractExpectedReturnsEstimator, ::Any, args...)
factory(me::AbstractExpectedReturnsEstimator, args...; kwargs...)
port_opt_view(alg::AbstractExpectedReturnsAlgorithm, ::Any, args...)
factory(alg::AbstractExpectedReturnsAlgorithm, args...; kwargs...)
robust_cov
robust_cor
moment_window_and_weights
windowed_preamble
demean_returns
```

## Full and semi moments

Moments other than the expected return can be estimated using the entire spectrum of deviations (full), or only the deviations below a target (semi/downside). These types allow us to provide such functionality.

```@docs
Full
Semi
```
