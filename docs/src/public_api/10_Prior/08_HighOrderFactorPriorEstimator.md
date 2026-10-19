```@meta
Description = "High order factor Prior estimator, public API of PortfolioOptimisers.jl: HighOrderFactorPriorEstimator, prior."
```

# High order factor Prior estimator

```@docs
HighOrderFactorPriorEstimator
prior(pe::HighOrderFactorPriorEstimator, X::MatNum, F::MatNum, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, kwargs...)
cokurtosis_processor
cokurtosis_expected_returns
coskewness_expected_returns
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
