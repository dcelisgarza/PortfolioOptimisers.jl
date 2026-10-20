```@meta
Description = "OWA risk measures constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# OWA risk measures constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-owa-risk-measures-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArray{<:Any, <:Any, <:ExactOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ExactOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArray{<:Any, <:Any, <:ApproxOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ApproxOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArray{<:Any, <:Any, <:SortingNetworkOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:SortingNetworkOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
