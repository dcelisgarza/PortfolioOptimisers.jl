```@meta
Description = "Relativistic XatRisk Constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# Relativistic XatRisk Constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-relativistic-xatrisk-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRiskRange, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticDrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
