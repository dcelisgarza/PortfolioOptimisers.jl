```@meta
Description = "Investable universe, private API of PortfolioOptimisers.jl: investable_reduction, coverage_reduction, expand_investable_weights, assert_internal_optimiser, …"
```

# Investable universe: private API

```@docs
investable_reduction(pr::AbstractPriorResult, opt::AbstractOptimisationEstimator, rd::ReturnsResult)
coverage_reduction(opt::AbstractOptimisationEstimator, rd::ReturnsResult)
expand_investable_weights
assert_internal_optimiser(::NonFiniteAllocationOptimisationResult)
assert_external_optimiser(::NonFiniteAllocationOptimisationResult)
non_investable_universe
```
