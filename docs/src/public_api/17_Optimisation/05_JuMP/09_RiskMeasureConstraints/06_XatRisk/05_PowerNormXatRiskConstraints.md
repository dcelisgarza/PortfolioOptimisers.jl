```@meta
Description = "Power-Norm XatRisk Constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# Power-Norm XatRisk Constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-power-norm-xatrisk-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormValueatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormValueatRiskRange, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::PowerNormDrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
