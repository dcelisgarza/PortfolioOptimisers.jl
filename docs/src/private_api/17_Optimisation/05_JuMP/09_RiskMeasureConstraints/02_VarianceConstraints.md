```@meta
Description = "Variance Constraints, private API of PortfolioOptimisers.jl: get_chol_or_sigma_pm, covariance_factor, chol_sigma_selector, …"
```

# Variance Constraints: private API

```@docs
get_chol_or_sigma_pm
covariance_factor
chol_sigma_selector
set_variance_risk_bounds_and_expression!
risk_contribution_constraints(r::Variance, opt::NonFRCJuMPOpt, pr::AbstractPriorResult)
set_risk!(model::JuMP.Model, i::Any, r::StandardDeviation, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::StandardDeviation, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance, opt::RiskBoundOwner, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance, opt::FactorRiskContribution, pr::AbstractPriorResult, ::Any, ::Any, b1::MatNum, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::UncertaintySetVariance, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; rd::ReturnsResult = ReturnsResult(), kwargs...)
sdp_rc_variance_flag!
sdp_variance_flag!
set_variance_risk!
set_sdp_variance_risk!
variance_risk_bounds_expr
rc_variance_constraints!
set_ucs_variance_risk!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
