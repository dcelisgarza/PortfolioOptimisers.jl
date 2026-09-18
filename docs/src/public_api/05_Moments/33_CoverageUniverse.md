```@meta
Description = "The Coverage Universe, public API of PortfolioOptimisers.jl: cov, cor, var, std, mean, coskewness, cokurtosis."
```

# [The Coverage Universe](@id api-coverage-universe)

An asset is in the **Coverage Universe** of one fit when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window. A prior reduces its returns matrix to that universe, fits every plain estimator on the clean block, and expands every block of its result onto the full asset universe with a `NaN` frame outside it.

The Asset Panel travels as the third positional argument of every moment verb, and the root method of each verb is that reduce-and-expand. A mask-aware estimator overrides its root and takes the whole window.

## The roots of the seven verbs

```@docs
cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cor(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
var(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
mean(me::AbstractExpectedReturnsEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
coskewness(ske::CoskewnessEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cokurtosis(kte::CokurtosisEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```

## The available-case seam

An estimator that carries a [`CoveragePolicy`](@ref) is a mask-aware estimator, so its panel method hands it the whole window and the panel's active mask instead of reducing to the Coverage Universe.

```@docs
mean(me::SimpleExpectedReturns, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
var(ve::SimpleVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ve::SimpleVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cov(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cor(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
var(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```
