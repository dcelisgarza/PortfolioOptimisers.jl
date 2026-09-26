```@meta
Description = "Median expected returns, public API of PortfolioOptimisers.jl: MedianExpectedReturns, mean."
```

# [Median expected returns](@id api-median-expected-returns)

```@docs
MedianExpectedReturns
mean(me::MedianExpectedReturns{Nothing}, X::MatNum; dims::Int = 1, kwargs...)
mean(me::MedianExpectedReturns{<:ObsWeights}, X::MatNum; dims::Int = 1, kwargs...)
```
