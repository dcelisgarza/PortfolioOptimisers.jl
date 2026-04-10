# Variance Constraints

```@docs
get_chol_or_sigma_pm
chol_sigma_selector
set_variance_risk_bounds_and_expression!
set_risk!(model::JuMP.Model, i::Any, r::StandardDeviation, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::StandardDeviation, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance, opt::NonFRCJuMPOpt, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance, opt::FactorRiskContribution, pr::AbstractPriorResult, ::Any, ::Any, b1::MatNum, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::UncertaintySetVariance, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; rd::ReturnsResult = ReturnsResult(), kwargs...)
sdp_rc_variance_flag!
sdp_variance_flag!
set_variance_risk!
set_sdp_variance_risk!
variance_risk_bounds_expr
variance_risk_bounds_val
rc_variance_constraints!
set_ucs_variance_risk!
```
