```@meta
Description = "OWA Risk Measure Constraints, private API of PortfolioOptimisers.jl: set_owa_constraints!, set_risk_constraints!."
```

# [OWA Risk Measure Constraints: private API](@id private-api-owa-risk-measure-constraints)

```@docs
set_owa_constraints!
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArray{<:Any, <:Any, <:ExactOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ExactOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArray{<:Any, <:Any, <:ApproxOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ApproxOrderedWeightsArray}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
