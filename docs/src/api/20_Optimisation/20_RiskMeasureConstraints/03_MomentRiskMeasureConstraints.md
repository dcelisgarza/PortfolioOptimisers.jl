# Moment Risk Constraints

```@docs
calc_risk_constraint_target
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:MeanAbsoluteDeviation}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:SecondMoment}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_second_moment_risk!
second_moment_bound_val
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:EvenMoment}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
```
