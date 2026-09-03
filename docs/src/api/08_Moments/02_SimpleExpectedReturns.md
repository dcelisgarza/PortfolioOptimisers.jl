# Simple expected returns

The most basic moment is the simple expected return. These types and functions implement it.

```@docs
SimpleExpectedReturns
show_fields(::SimpleExpectedReturns)
mean(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
```

## Incremental fit

The sample mean folds one observation at a time, so a long history need not be held or re-read. [`partial_fit!`](@ref) returns a new estimator whose `cache` field carries the state, and `mean` reads the fit off the estimator alone.

```@docs
SimpleExpectedReturnsState
partial_fit!(state::SimpleExpectedReturnsState, x::VecNum)
partial_fit!(me::SimpleExpectedReturns, X::MatNum; dims::Int = 1)
partial_fit!(me::SimpleExpectedReturns, x::VecNum)
expected_returns_state_seed
mean(me::SimpleExpectedReturns, state::SimpleExpectedReturnsState)
merge_states(a::SimpleExpectedReturnsState, b::SimpleExpectedReturnsState)
```
