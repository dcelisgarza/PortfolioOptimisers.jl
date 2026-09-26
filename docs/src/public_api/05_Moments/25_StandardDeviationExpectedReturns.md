```@meta
Description = "Standard deviation expected returns, public API of PortfolioOptimisers.jl: StandardDeviationExpectedReturns, VarianceExpectedReturns, mean, factory."
```

# [Standard deviation expected returns](@id api-standard-deviation-expected-returns)

```@docs
StandardDeviationExpectedReturns
VarianceExpectedReturns
mean(me::StandardDeviationExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
factory(ce::VarianceExpectedReturns, args...; kwargs...)
mean(me::VarianceExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
```
