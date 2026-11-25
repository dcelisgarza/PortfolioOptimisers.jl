```@meta
Description = "Position constraint generation, public API of PortfolioOptimisers.jl: PositionConstraintEstimator, PositionConstraint, linear_constraints, port_opt_view."
```

# Position constraint generation

```@docs
PositionConstraintEstimator
PositionConstraint
linear_constraints(pce::PositionConstraintEstimator, sets::Option{<:UniverseSets}; kwargs...)
linear_constraints(lcs::VecPcE_Pc_EcE_LcE_Lc, sets::Option{<:UniverseSets}; kwargs...)
port_opt_view(pce::PositionConstraintEstimator, i, args...)
```
