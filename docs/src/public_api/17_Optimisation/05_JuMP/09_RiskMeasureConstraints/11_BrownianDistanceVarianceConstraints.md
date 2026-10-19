```@meta
Description = "Brownian Distance Variance Constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# Brownian Distance Variance Constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-brownian-distance-variance-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, ::Any, r::BrownianDistanceVariance, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
