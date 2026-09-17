```@meta
Description = "Entropic XatRisk Constraints, private API of PortfolioOptimisers.jl: set_risk_constraints!, set_entropic_risk_constraints!."
```

# Entropic XatRisk Constraints: private API

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRisk, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRiskRange, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicDrawdownatRisk, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_entropic_risk_constraints!
```
