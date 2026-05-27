# Simple expected returns

The most basic moment is the simple expected return. These types and functions implement it.

```@docs
SimpleExpectedReturns
factory(me::SimpleExpectedReturns, w::ObsWeights)
mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
```
