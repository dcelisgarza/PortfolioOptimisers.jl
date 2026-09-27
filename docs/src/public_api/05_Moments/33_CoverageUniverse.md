```@meta
Description = "The Coverage Universe, public API of PortfolioOptimisers.jl: cov, cor, var, std, mean, coskewness, cokurtosis."
```

# [The Coverage Universe](@id api-coverage-universe)

An asset is in the coverage universe of one fit when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window. A prior keeps only those assets of its returns matrix and fits each estimator on the smaller matrix. It then puts every result back on the full set of assets, with `NaN` for each asset outside the coverage universe.

Each of the seven moment functions takes the asset panel as its third positional argument. The default method of each function removes the assets outside the coverage universe and puts them back as `NaN`. An estimator that reads the masks itself has its own method, and receives the whole window.

## The default methods of the seven moment functions

```@docs
cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cor(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
var(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
mean(me::AbstractExpectedReturnsEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
coskewness(ske::CoskewnessEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cokurtosis(kte::CokurtosisEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```

## Estimators that fit each asset on its own observations

An estimator that carries a [`CoveragePolicy`](@ref) reads the masks itself. Its method with a panel passes it the whole window and the active mask of the panel, and does not remove the assets outside the coverage universe.

```@docs
mean(me::SimpleExpectedReturns, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
var(ve::SimpleVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ve::SimpleVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cov(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cor(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
var(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```
