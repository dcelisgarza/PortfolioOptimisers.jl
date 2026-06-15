# Median expected returns

```@docs
MedianExpectedReturns
mean(me::MedianExpectedReturns{Nothing}, X::MatNum; dims::Int = 1,
                         kwargs...)
mean(me::MedianExpectedReturns{<:ObsWeights}, X::MatNum; dims::Int = 1,
                         kwargs...)
```
