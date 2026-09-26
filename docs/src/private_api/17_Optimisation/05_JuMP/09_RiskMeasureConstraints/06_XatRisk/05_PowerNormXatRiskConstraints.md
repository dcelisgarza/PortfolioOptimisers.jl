```@meta
Description = "Power-Norm XatRisk Constraints, private API of PortfolioOptimisers.jl: set_risk_constraints!, set_power_norm_risk_constraints!."
```

# [Power-Norm XatRisk Constraints: private API](@id private-api-power-norm-xatrisk-constraints)

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormValueatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormValueatRiskRange, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormDrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_power_norm_risk_constraints!
```
