```@meta
Description = "Tracking Risk Measure Constraints, private API of PortfolioOptimisers.jl: set_risk_constraints!, set_tracking_risk!, set_risk_tr_constraints!, …"
```

# Tracking Risk Measure Constraints: private API

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:L1Norm}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:Union{<:L2Norm, <:SquaredL2Norm}}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:LpNorm}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:LInfNorm}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any, <:IndependentVariableTracking}, opt::RiskConstraintOwner, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any, <:DependentVariableTracking}, opt::RiskConstraintOwner, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...; kwargs...)
set_tracking_risk!
set_risk_tr_constraints!
set_risk_tracking_risk_constraints!
```
