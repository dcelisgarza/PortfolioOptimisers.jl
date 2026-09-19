```@meta
Description = "Negative Skewness Constraints, private API of PortfolioOptimisers.jl: get_chol_or_V_pm, set_negative_skewness_risk!, set_risk_constraints!."
```

# Negative Skewness Constraints: private API

```@docs
get_chol_or_V_pm
set_negative_skewness_risk!
set_risk_constraints!(model::JuMP.Model, i::Any, r::NegativeSkewness, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
