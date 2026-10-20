```@meta
Description = "Tracking Risk Measure Constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# Tracking Risk Measure Constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-tracking-risk-measure-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:L1Norm}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:Union{<:L2Norm, <:SquaredL2Norm}}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:LpNorm}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:LInfNorm}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any, <:IndependentVariableTracking}, opt::RiskConstraintOwner, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any, <:DependentVariableTracking}, opt::RiskConstraintOwner, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...; kwargs...)
```
