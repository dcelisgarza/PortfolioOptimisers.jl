```@meta
Description = "Constraint Block, public API of PortfolioOptimisers.jl: ConstraintBlock, ResolvedConstraintBlock, factory, port_opt_view, needs_previous_weights, …"
```

# Constraint Block

```@docs
ConstraintBlock
ResolvedConstraintBlock
factory(cbe::ConstraintBlock, w::AbstractVector)
port_opt_view(cbe::ConstraintBlock, i, args...)
needs_previous_weights(cbe::ConstraintBlock)
obs_weights_view(cbe::ConstraintBlock, i)
```
