# Tracking Risk Measure Constraints

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:L1Norm}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:Union{<:L2Norm, <:SquaredL2Norm}}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:LpNorm}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:LInfNorm}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any, <:IndependentVariableTracking}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any, <:DependentVariableTracking}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...; kwargs...)
set_tracking_risk!
set_risk_tr_constraints!
set_risk_tracking_risk_constraints!
```
<!-- set_triv_risk_constraints! -->
<!-- set_trdv_risk_constraints! -->
