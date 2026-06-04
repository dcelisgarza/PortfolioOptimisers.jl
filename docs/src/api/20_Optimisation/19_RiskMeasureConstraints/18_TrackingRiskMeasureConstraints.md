# Tracking Risk Measure Constraints

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:L1Tracking}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:Union{<:L2Tracking, <:SquaredL2Tracking}}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:LpTracking}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::TrackingRiskMeasure{<:Any, <:Any, <:LInfTracking}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any, <:IndependentVariableTracking}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any, <:DependentVariableTracking}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...; kwargs...)
set_tracking_risk!
set_risk_tr_constraints!
set_triv_risk_constraints!
```
<!-- set_trdv_risk_constraints! -->
