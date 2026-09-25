```@meta
Description = "XatRisk Constraints, private API of PortfolioOptimisers.jl: set_risk_constraints!, set_mip_quantile_risk_constraints!, mip_series_spread, …"
```

# XatRisk Constraints: private API

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRisk{<:Any, <:Any, <:Any, <:MIPValueatRisk}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:MIPValueatRisk}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:DistributionValueatRisk}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::DrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_mip_quantile_risk_constraints!
mip_series_spread
mip_fees_keep_spread
mip_big_m
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
