```@meta
Description = "The Coverage Universe, private API of PortfolioOptimisers.jl: coverage_mask, coverage_sentinel, coverage_reduction, coverage_reduced_pair, …"
```

# The Coverage Universe: private API

The coverage universe of a fit is the set of assets whose returns are finite, and whose entries in the active mask of the [`AssetPanel`](@ref) are `true`, on every row of the window. A prior fits each ordinary moment estimator on the returns of those assets alone. It then writes the results into arrays over all the assets, where the entries of every other asset are `NaN`.

Every moment function takes the asset panel as its third positional argument. Its panel method does this reduction and expansion. An estimator that reads the active mask itself replaces that method, and takes the whole window.

## The universe and the reduction

```@docs
PortfolioOptimisers.coverage_mask
PortfolioOptimisers.coverage_sentinel
PortfolioOptimisers.coverage_reduction(X::MatNum, pnl::Option{<:AssetPanel})
PortfolioOptimisers.coverage_reduction(rd::AbstractReturnsResult)
PortfolioOptimisers.coverage_reduced_pair
PortfolioOptimisers.panel_moment_masks
```

## The expansion

```@docs
PortfolioOptimisers.coverage_nan_frame
PortfolioOptimisers.coverage_pair_index
PortfolioOptimisers.expand_columns
PortfolioOptimisers.reduce_columns
PortfolioOptimisers.expand_rows
PortfolioOptimisers.expand_vector
PortfolioOptimisers.expand_moment
PortfolioOptimisers.expand_regression
PortfolioOptimisers.expand_idiosyncratic_covariance
```

## The check for missing returns

```@docs
PortfolioOptimisers.assert_finite_sample
```

## The panel method of `variance_series`

```@docs
PortfolioOptimisers.variance_series(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
```

## The available-case fit

An estimator with a [`CoveragePolicy`](@ref) fits each asset, or each pair of assets, on the rows where its returns are finite and active. Its panel method therefore passes the whole window and the active mask to the estimator, and does not reduce the window to the coverage universe.

```@docs
PortfolioOptimisers.coverage_panel_moment
PortfolioOptimisers.coverage_panel_moment(f, est, ::Nothing, X::MatNum, pnl::Option{<:AssetPanel}, expand)
PortfolioOptimisers.coverage_panel_moment(f, est, ::CoveragePolicy, X::MatNum, pnl::Option{<:AssetPanel}, ::Any)
PortfolioOptimisers.coverage_variance_series
PortfolioOptimisers.coverage_variance_series(ce::AbstractCovarianceEstimator, ::Nothing, X::MatNum, pnl::Option{<:AssetPanel})
PortfolioOptimisers.coverage_variance_series(ce::AbstractCovarianceEstimator, ::CoveragePolicy, X::MatNum, pnl::Option{<:AssetPanel})
PortfolioOptimisers.variance_series(ce::Covariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
PortfolioOptimisers.variance_series(ve::SimpleVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
PortfolioOptimisers.coverage_floor
```
