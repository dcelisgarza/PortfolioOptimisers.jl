```@meta
Description = "Relativistic XatRisk Constraints, private API of PortfolioOptimisers.jl: set_risk_constraints!, set_relativistic_risk_constraints!."
```

# [Relativistic XatRisk Constraints: private API](@id private-api-relativistic-xatrisk-constraints)

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRiskRange, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticDrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_relativistic_risk_constraints!
```
