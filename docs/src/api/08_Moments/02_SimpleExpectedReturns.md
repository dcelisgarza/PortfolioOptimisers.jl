# Simple expected returns

The most basic moment is the simple expected return. These types and functions implement it.

```@docs
SimpleExpectedReturns
factory(me::SimpleExpectedReturns, w::StatsBase.AbstractWeights)
mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
```
