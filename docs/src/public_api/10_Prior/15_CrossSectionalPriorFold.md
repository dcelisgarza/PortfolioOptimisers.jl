```@meta
Description = "Cross-Sectional Prior Fold, public API of PortfolioOptimisers.jl: partial_fit!, prior, merge_states, port_opt_view, carry_folds."
```

# Cross-Sectional Prior Fold

A [`CrossSectionalFactorPrior`](@ref) that [`Online`](@ref) does not wrap updates with each new block of observations. Its first `partial_fit!` creates a [`PortfolioOptimisers.CrossSectionalCarryState`](@ref). Each later update computes the exposures, the regression and the idiosyncratic variance of the new observations alone, from the last panel rows that its descriptors read. Then `prior(pe)` builds the result with the code of the batch fit.

```@docs
PortfolioOptimisers.partial_fit!(pe::CrossSectionalFactorPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:PortfolioOptimisers.Option{<:PortfolioOptimisers.CrossSectionalCarryState}}, rd::ReturnsResult)
prior(pe::CrossSectionalFactorPrior, st::PortfolioOptimisers.CrossSectionalCarryState; strict::Bool = false, kwargs...)
merge_states(::PortfolioOptimisers.CrossSectionalCarryState, ::PortfolioOptimisers.CrossSectionalCarryState)
PortfolioOptimisers.port_opt_view(::PortfolioOptimisers.CrossSectionalCarryState, i, args...)
```

The carry folds its factor prior one row at a time when [`PortfolioOptimisers.carry_folds`](@ref) answers `true` for it. A user prior implements that verb to fold on the carry and to pass [`FoldOnly`](@ref).

```@docs
PortfolioOptimisers.carry_folds
```
