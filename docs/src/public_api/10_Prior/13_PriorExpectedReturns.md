```@meta
Description = "The Prior adapter for an expected-returns slot, public API of PortfolioOptimisers.jl: PriorExpectedReturns, mean, partial_fit!."
```

# [The Prior adapter for an expected-returns slot](@id api-the-prior-adapter-for-an-expected-returns-slot)

`PriorExpectedReturns` wraps a prior estimator, so that any field that takes an expected returns estimator, `me`, can hold a prior. It fits the prior on the returns and keeps only its expected returns, `mu`. A Black-Litterman or shrunk mean can then feed a mean reversion rule, an expected return risk measure or an empirical prior. An expected returns estimator never receives factor returns, so the constructor throws an error for a prior that needs them.

```@docs
PriorExpectedReturns
mean(me::PriorExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
partial_fit!(me::PriorExpectedReturns, X::VecNum_MatNum; kwargs...)
```
