```@meta
Description = "Relativistic XatRisk Constraints, private API of PortfolioOptimisers.jl: set_risk_constraints!, set_relativistic_risk_constraints!."
```

# Relativistic XatRisk Constraints: private API

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRisk, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRiskRange, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticDrawdownatRisk, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_relativistic_risk_constraints!
```
