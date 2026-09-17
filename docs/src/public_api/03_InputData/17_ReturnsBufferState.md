```@meta
Description = "Returns buffer state, public API of PortfolioOptimisers.jl: partial_fit!, port_opt_view."
```

# Returns buffer state

## The fold context of the online step

An optimiser's online step forwards each observation to its prior and records the rest of the
carrier in a [`PortfolioOptimisers.ReturnsBufferState`](@ref): the factor, benchmark and
timestamp columns as buffers of their own, and the names and the static Asset Panel pinned by
the first step. The returns are owned once, by the prior, and this state holds them only where
no prior sits beneath the optimiser. [`PortfolioOptimisers.returns_result`](@ref) rebuilds the
[`ReturnsResult`](@ref) a batch fit over the same observations would have read.

```@docs
PortfolioOptimisers.partial_fit!(state::PortfolioOptimisers.ReturnsBufferState, rd::ReturnsResult; own_returns::Bool = false)
PortfolioOptimisers.port_opt_view(x::PortfolioOptimisers.ReturnsBufferState, i, args...)
```
