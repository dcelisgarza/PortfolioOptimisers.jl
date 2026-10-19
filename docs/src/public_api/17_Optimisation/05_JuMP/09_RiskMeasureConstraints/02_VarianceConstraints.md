```@meta
Description = "Variance Constraints, public API of PortfolioOptimisers.jl: variance_risk_bounds_val, set_risk_constraints!."
```

# Variance Constraints

```@docs
variance_risk_bounds_val
set_risk_constraints!(model::JuMP.Model, i::Any, r::StandardDeviation, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance, opt::RiskBoundOwner, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance, opt::FactorRiskContribution, pr::AbstractPriorResult, ::Any, ::Any, b1::MatNum, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::UncertaintySetVariance, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; rd::ReturnsResult = ReturnsResult(), kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
