```@meta
Description = "JuMP weight finaliser, public API of PortfolioOptimisers.jl: JuMPWeightFinaliser, set_clustering_weight_finaliser_alg!."
```

# JuMP weight finaliser

The weight finaliser below moves weights into their bounds with a JuMP programme. It can hold a [`ConstraintBlock`](@ref), so an optimiser with no solver of its own, such as [`HierarchicalRiskParity`](@ref) or [`EqualWeighted`](@ref), takes the constraint kinds of a [`JuMPOptimiser`](@ref).

```@docs
JuMPWeightFinaliser
set_clustering_weight_finaliser_alg!
```
