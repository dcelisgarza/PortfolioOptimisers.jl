```@meta
Description = "Returns buffer state, public API of PortfolioOptimisers.jl: partial_fit!, port_opt_view, merge_states."
```

# Returns buffer state

## What an optimiser stores between updates

When you update an optimiser with `partial_fit!`, it passes the asset returns to its prior. It
stores the rest of the [`ReturnsResult`](@ref) in a [`PortfolioOptimisers.ReturnsBufferState`](@ref).
The benchmark returns and the timestamps each go in a buffer of their own. The factor returns go in
a buffer here only when the prior does not read them, or when the optimiser has no prior. The state
also stores the asset names and the static asset panel of the first update. The prior holds
the asset returns, and this state stores them only when the optimiser has no prior.
[`PortfolioOptimisers.returns_result`](@ref) rebuilds the `ReturnsResult` that a batch fit over the
same observations would read.

```@docs
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.ReturnsBufferState, rd::ReturnsResult; own_returns::Bool = false)
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.ReturnsBufferState, i, args...)
merge_states(a::PortfolioOptimisers.ReturnsBufferState, b::PortfolioOptimisers.ReturnsBufferState)
```
