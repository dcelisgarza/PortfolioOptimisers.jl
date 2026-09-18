```@meta
Description = "Variance Skew Kurtosis, private API of PortfolioOptimisers.jl: bigger_is_better, resolve_deferred_quantities, supports_precomputed_returns."
```

# Variance Skew Kurtosis: private API

```@docs
bigger_is_better(::Skewness)
resolve_deferred_quantities(r::VarianceSkewKurtosis, pr::AbstractPriorResult)
supports_precomputed_returns(r::Skewness)
supports_precomputed_returns(::VarianceSkewKurtosis)
```
