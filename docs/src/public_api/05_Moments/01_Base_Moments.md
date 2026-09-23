```@meta
Description = "Base moments, public API of PortfolioOptimisers.jl: AbstractCovarianceEstimator, AbstractVarianceEstimator, AbstractExpectedReturnsEstimator, port_opt_view, …"
```

# Base moments

## Abstract moment types and fallbacks

The abstract types below are the supertypes of the estimators of the expected returns, the variance and the covariance. The functions below are their default methods, which an estimator uses when it does not define its own.

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

## Full and semi moments

A moment other than the expected return can use every deviation from the target, `FullMoment`, or only the deviations below the target, `SemiMoment`. An estimator that has both forms takes one of these two types in its `alg` field.

```@docs
FullMoment
SemiMoment
```
