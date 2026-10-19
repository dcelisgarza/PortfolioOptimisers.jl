```@meta
Description = "XatRisk Constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# XatRisk Constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-xatrisk-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRisk{<:Any, <:Any, <:Any, <:MIPValueatRisk}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:MIPValueatRisk}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:DistributionValueatRisk}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::DrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
