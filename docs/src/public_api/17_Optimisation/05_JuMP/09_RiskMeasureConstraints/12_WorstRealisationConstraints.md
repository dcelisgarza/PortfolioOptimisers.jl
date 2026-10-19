```@meta
Description = "Worst Realisation Constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# Worst Realisation Constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-worst-realisation-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, ::Any, r::WorstRealisation, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
