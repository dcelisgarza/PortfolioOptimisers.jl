# Moment Risk Constraints

```@docs
calc_risk_constraint_target
set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
set_second_moment_risk!
second_moment_bound_val
```
