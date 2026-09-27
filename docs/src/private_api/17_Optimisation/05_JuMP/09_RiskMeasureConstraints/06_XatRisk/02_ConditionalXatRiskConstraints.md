```@meta
Description = "Conditional XatRisk Constraints, private API of PortfolioOptimisers.jl: set_risk_constraints!, set_conditional_risk_constraints!, …"
```

# [Conditional XatRisk Constraints: private API](@id private-api-conditional-xatrisk-constraints)

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalValueatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalValueatRiskRange, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::DistributionallyRobustConditionalValueatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::DistributionallyRobustConditionalValueatRiskRange, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalDrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::DistributionallyRobustConditionalDrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_conditional_risk_constraints!
set_dr_conditional_risk_constraints!
```
