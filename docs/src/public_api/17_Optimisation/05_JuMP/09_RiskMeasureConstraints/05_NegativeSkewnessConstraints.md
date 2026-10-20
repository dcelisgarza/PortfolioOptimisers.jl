```@meta
Description = "Negative Skewness Constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# Negative Skewness Constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-negative-skewness-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::NegativeSkewness, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
