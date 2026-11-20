```@meta
Description = "Cross-Sectional Prior Fold (a), public API of PortfolioOptimisers.jl: merge_states, port_opt_view, carry_folds."
```

# Cross-Sectional Prior Fold (a)

A [`CrossSectionalFactorPrior`](@ref) that [`Online`](@ref) does not wrap updates with each new block of observations. Its first `partial_fit!` creates a [`PortfolioOptimisers.CrossSectionalCarryState`](@ref). The state merges with another state and takes a view on its assets as the methods below state.

```@docs
merge_states(::PortfolioOptimisers.CrossSectionalCarryState, ::PortfolioOptimisers.CrossSectionalCarryState)
PortfolioOptimisers.port_opt_view(::PortfolioOptimisers.CrossSectionalCarryState, i, args...)
```

The carry folds its factor prior one row at a time when [`PortfolioOptimisers.carry_folds`](@ref) answers `true` for it. A user prior implements that verb to fold on the carry and to pass [`FoldOnly`](@ref).

```@docs
PortfolioOptimisers.carry_folds
```
