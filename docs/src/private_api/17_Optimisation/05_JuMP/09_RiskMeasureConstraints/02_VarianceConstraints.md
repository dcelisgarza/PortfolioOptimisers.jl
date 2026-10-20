```@meta
Description = "Variance Constraints, private API of PortfolioOptimisers.jl: get_chol_or_sigma_pm, chol_sigma_selector, set_variance_risk_bounds_and_expression!, …"
```

# Variance Constraints: private API

```@docs
get_chol_or_sigma_pm
chol_sigma_selector
set_variance_risk_bounds_and_expression!
risk_contribution_constraints(r::Variance, opt::NonFRCJuMPOpt, pr::AbstractPriorResult)
set_risk!(model::JuMP.Model, i::Any, r::StandardDeviation, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
sdp_rc_variance_flag!
sdp_variance_flag!
set_variance_risk!
set_sdp_variance_risk!
variance_risk_bounds_expr
rc_variance_constraints!
set_ucs_variance_risk!
sdp_lifted_set
ucs_variance_risk_bounds!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
