```@meta
Description = "The Prior adapter for an expected-returns slot, public API of PortfolioOptimisers.jl: PriorExpectedReturns, mean, partial_fit!."
```

# The Prior adapter for an expected-returns slot

The library-wide adapter that lets any `me` slot hold a prior estimator and read its `mu` alone: a Black–Litterman or shrunk mean may drive a reversion step, an expected-return risk measure or an empirical prior. A prior that requires factor returns is refused at construction, because the expected-returns seam carries none.

```@docs
PriorExpectedReturns
mean(me::PriorExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
partial_fit!(me::PriorExpectedReturns, X::VecNum_MatNum; kwargs...)
```
