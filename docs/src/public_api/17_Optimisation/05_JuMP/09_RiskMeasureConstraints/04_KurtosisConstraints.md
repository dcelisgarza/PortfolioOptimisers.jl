```@meta
Description = "Kurtosis Constraints, public API of PortfolioOptimisers.jl: set_risk_constraints!."
```

# Kurtosis Constraints

The methods below are public. The other names of this topic are private, and the [private page](@ref private-api-kurtosis-constraints) documents them.

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Integer, <:Any, <:Any}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::Kurtosis{<:Any, <:Any, <:Any, <:Any, Nothing, <:Any, <:Any}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
