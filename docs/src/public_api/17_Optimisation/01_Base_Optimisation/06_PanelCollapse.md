```@meta
Description = "Panel collapse, public API of PortfolioOptimisers.jl: calc_net_returns, RenormaliseActive, InactiveAsCash."
```

# Panel collapse

```@docs
calc_net_returns(res::OptimisationResult, X::MatNum, fees::Option{<:Fees} = nothing, wd::Option{<:AbstractWeightDrift} = nothing, obs = nothing)
RenormaliseActive
InactiveAsCash
```
