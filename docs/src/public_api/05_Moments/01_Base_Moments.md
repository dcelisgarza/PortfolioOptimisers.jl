```@meta
Description = "Base moments, public API of PortfolioOptimisers.jl: AbstractCovarianceEstimator, AbstractVarianceEstimator, AbstractExpectedReturnsEstimator, port_opt_view, …"
```

# Base moments

## Abstract moment types and fallbacks

Some optimisations and constraints make use of summary statistics. These types and functions form the base for moment estimation in `PortfolioOptimisers.jl`.

They also provide generic fallbacks for the various functionality in the library.

```@docs
AbstractCovarianceEstimator
AbstractVarianceEstimator
AbstractExpectedReturnsEstimator
port_opt_view(ce::StatsBase.CovarianceEstimator, ::Any, args...)
factory(ce::StatsBase.CovarianceEstimator, args...)
port_opt_view(me::AbstractExpectedReturnsEstimator, ::Any, args...)
factory(me::AbstractExpectedReturnsEstimator, args...; kwargs...)
port_opt_view(alg::AbstractExpectedReturnsAlgorithm, ::Any, args...)
factory(alg::AbstractExpectedReturnsAlgorithm, args...; kwargs...)
cov(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
cov(ce::AbstractCovarianceEstimator, state::SampleBufferState)
cor(ce::AbstractCovarianceEstimator, state::SampleBufferState)
var(ve::AbstractVarianceEstimator, state::SampleBufferState)
std(ve::AbstractVarianceEstimator, state::SampleBufferState)
mean(me::AbstractExpectedReturnsEstimator, state::SampleBufferState)
```

## FullMoment and semi moments

Moments other than the expected return can be estimated using the entire spectrum of deviations (full), or only the deviations below a target (semi/downside). These types allow us to provide such functionality.

```@docs
FullMoment
SemiMoment
```
