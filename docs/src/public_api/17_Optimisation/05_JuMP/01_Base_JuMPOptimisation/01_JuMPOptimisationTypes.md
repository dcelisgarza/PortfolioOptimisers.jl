```@meta
Description = "JuMP optimisation types, public API of PortfolioOptimisers.jl: CustomJuMPConstraint, VecJuMPConstr, CustomJuMPObjective, VecJuMPObj, …"
```

# JuMP optimisation types

```@docs
CustomJuMPConstraint
VecJuMPConstr
CustomJuMPObjective
VecJuMPObj
JuMPOptimisationSolution
Base.propertynames(r::RJR_NRJR)
Base.getproperty(r::RJR_NRJR, sym::Symbol)
add_custom_objective_term!
add_custom_constraint!
needs_previous_weights(::CustomJuMPConstraint)
needs_previous_weights(::CustomJuMPObjective)
```
