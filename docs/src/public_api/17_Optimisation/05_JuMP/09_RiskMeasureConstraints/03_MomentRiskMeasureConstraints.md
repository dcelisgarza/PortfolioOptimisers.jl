```@meta
Description = "Moment risk measure constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# Moment risk measure constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-moment-risk-measure-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:MeanAbsoluteDeviation}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:SecondMoment}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::LowOrderMoment{<:Any, <:Any, <:Any, <:EvenMoment}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
