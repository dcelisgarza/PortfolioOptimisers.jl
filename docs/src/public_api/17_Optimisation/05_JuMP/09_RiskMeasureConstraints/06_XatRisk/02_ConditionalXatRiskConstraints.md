```@meta
Description = "Conditional XatRisk Constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# Conditional XatRisk Constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-conditional-xatrisk-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalValueatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalValueatRiskRange, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::DistributionallyRobustConditionalValueatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::DistributionallyRobustConditionalValueatRiskRange, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalDrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::DistributionallyRobustConditionalDrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
