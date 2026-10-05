```@meta
Description = "Panel collapse, public API of PortfolioOptimisers.jl: calc_net_returns, AbstractPanelCollapseAlgorithm, active_weight_divisor, RenormaliseActive, …"
```

# Panel collapse

```@docs
calc_net_returns(res::OptimisationResult, X::MatNum, fees::Option{<:Fees} = nothing, wd::Option{<:AbstractWeightDrift} = nothing, obs = nothing)
PortfolioOptimisers.AbstractPanelCollapseAlgorithm
PortfolioOptimisers.active_weight_divisor
RenormaliseActive
InactiveAsCash
```
