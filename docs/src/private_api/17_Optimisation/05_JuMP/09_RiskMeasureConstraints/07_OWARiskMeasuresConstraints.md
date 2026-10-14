```@meta
Description = "OWA risk measures constraints, private API of PortfolioOptimisers.jl: set_owa_constraints!, set_risk_constraints!, set_sorting_network_owa!."
```

# [OWA risk measures constraints: private API](@id private-api-owa-risk-measures-constraints)

```@docs
set_owa_constraints!
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArray{<:Any, <:Any, <:ExactOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ExactOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArray{<:Any, <:Any, <:ApproxOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ApproxOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_sorting_network_owa!
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArray{<:Any, <:Any, <:SortingNetworkOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:SortingNetworkOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
