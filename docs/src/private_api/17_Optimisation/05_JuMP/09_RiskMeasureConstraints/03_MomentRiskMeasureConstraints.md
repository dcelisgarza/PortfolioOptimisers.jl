```@meta
Description = "Moment risk measure constraints, private API of PortfolioOptimisers.jl: calc_risk_constraint_target, set_risk_constraints!, set_second_moment_risk!, …"
```

# [Moment risk measure constraints: private API](@id private-api-moment-risk-measure-constraints)

```@docs
calc_risk_constraint_target
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:MeanAbsoluteDeviation}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:SecondMoment}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_second_moment_risk!
second_moment_bound_val
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:EvenMoment}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
