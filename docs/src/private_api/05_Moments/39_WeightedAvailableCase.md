```@meta
Description = "Weighted available case, private API of PortfolioOptimisers.jl: weighted_coverage_centres, coverage_mean, weighted_observed_variance, coverage_variance, …"
```

# Weighted available case: private API

With a [`CoveragePolicy`](@ref) and observation weights, an estimator fits each asset, each pair of assets and each cell of a higher-order co-moment over its own valid observations, with the weights of those observations. A weighted fit does not fold, because the weights describe the whole sample, so these arms make one pass over the block. Each fit drops its other observations with their weights, through [`PortfolioOptimisers.observed_sample`](@ref). The unweighted arms fold, and they stay in the file of each estimator.

```@docs
PortfolioOptimisers.weighted_coverage_centres
PortfolioOptimisers.coverage_mean(me::SimpleExpectedReturns{<:PortfolioOptimisers.ObsWeights}, cvg::CoveragePolicy, X::PortfolioOptimisers.MatNum)
PortfolioOptimisers.weighted_observed_variance
PortfolioOptimisers.coverage_variance(f, ve::SimpleVariance{<:Any, <:PortfolioOptimisers.ObsWeights}, cvg::CoveragePolicy, ::PortfolioOptimisers.AbstractExpectedReturnsEstimator, X::PortfolioOptimisers.MatNum)
PortfolioOptimisers.coverage_variance(f, ve::SimpleVariance{<:Any, <:PortfolioOptimisers.ObsWeights}, cvg::CoveragePolicy, X::PortfolioOptimisers.VecNum)
PortfolioOptimisers.weighted_correlation_estimator
PortfolioOptimisers.coverage_inner_corrected
PortfolioOptimisers.weighted_pair_sample
PortfolioOptimisers.weighted_pair_covariance
PortfolioOptimisers.coverage_covariance(f, ce::Covariance{<:Any, <:Any, <:FullMoment, <:PortfolioOptimisers.ObsWeights}, cvg::CoveragePolicy, X::PortfolioOptimisers.MatNum)
PortfolioOptimisers.weighted_comoment_cell
PortfolioOptimisers.weighted_comoment
PortfolioOptimisers.weighted_comoment_block
```
