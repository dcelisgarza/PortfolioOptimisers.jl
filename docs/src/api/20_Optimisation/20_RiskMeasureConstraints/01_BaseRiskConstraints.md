# Base Risk Constraints

```@docs
scalarise_risk_expression!
set_risk_constraints!(model::JuMP.Model, r::RiskMeasure, opt::JuMPOptimisationEstimator, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...; kwargs...)
set_risk_upper_bound!
set_risk_expression!
set_risk_bounds_and_expression!
set_drawdown_constraints!
NonFRCJuMPOpt
```
