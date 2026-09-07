# [The Coverage Universe](@id api-coverage-universe)

An asset is in the **Coverage Universe** of one fit when its return is finite and the active mask of the [`AssetPanel`](@ref) is `true` at every row of the window. A prior reduces its returns matrix to that universe, fits every plain estimator on the clean block, and expands every block of its result onto the full asset universe with a `NaN` frame outside it.

The Asset Panel travels as the third positional argument of every moment verb, and the root method of each verb is that reduce-and-expand. A mask-aware estimator overrides its root and takes the whole window.

## The universe and the reduction

```@docs
PortfolioOptimisers.coverage_mask
PortfolioOptimisers.coverage_sentinel
PortfolioOptimisers.coverage_reduction
PortfolioOptimisers.coverage_reduced_pair
PortfolioOptimisers.panel_moment_masks
```

## The expansion

```@docs
PortfolioOptimisers.coverage_nan_frame
PortfolioOptimisers.coverage_pair_index
PortfolioOptimisers.expand_columns
PortfolioOptimisers.expand_rows
PortfolioOptimisers.expand_vector
PortfolioOptimisers.expand_moment
PortfolioOptimisers.expand_regression
PortfolioOptimisers.expand_idiosyncratic_covariance
```

## The refusal

```@docs
PortfolioOptimisers.assert_finite_sample
```

## The roots of the seven verbs

```@docs
cov(ce::AbstractCovarianceEstimator, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
cor(ce::AbstractCovarianceEstimator, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
var(ce::AbstractCovarianceEstimator, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
std(ce::AbstractCovarianceEstimator, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
mean(me::AbstractExpectedReturnsEstimator, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
coskewness(ske::CoskewnessEstimator, X::MatNum, pnl::Option{<:AssetPanel};
                    dims::Int = 1, kwargs...)
cokurtosis(kte::CokurtosisEstimator, X::MatNum, pnl::Option{<:AssetPanel};
                    dims::Int = 1, kwargs...)
```
