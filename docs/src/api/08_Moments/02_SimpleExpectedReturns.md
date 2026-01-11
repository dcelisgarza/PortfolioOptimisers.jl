# Simple expected returns

The most basic moment is the simple expected return. These types and functions implement it.

```@docs
SimpleExpectedReturns
mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
factory(me::SimpleExpectedReturns, w::Option{<:StatsBase.AbstractWeights} = nothing)
```
