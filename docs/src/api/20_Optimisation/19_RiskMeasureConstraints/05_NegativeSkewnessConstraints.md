# Negative Skewness Constraints

```@docs
get_chol_or_V_pm
set_negative_skewness_risk!
set_risk_constraints!(model::JuMP.Model, i::Any, r::NegativeSkewness, opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior, args...; kwargs...)
set_risk_constraints!(::JuMP.Model, ::Any, ::NegativeSkewness, ::RiskJuMPOptimisationEstimator, pr::LowOrderPrior, args...; kwargs...)
```
