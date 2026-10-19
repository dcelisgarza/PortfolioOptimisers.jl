```@meta
Description = "Entropic XatRisk Constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# Entropic XatRisk Constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-entropic-xatrisk-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicValueatRiskRange, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::EntropicDrawdownatRisk, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
