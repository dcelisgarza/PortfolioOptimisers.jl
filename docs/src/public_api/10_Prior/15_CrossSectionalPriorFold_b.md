```@meta
Description = "Cross-Sectional Prior Fold (b), public API of PortfolioOptimisers.jl: prior."
```

# Cross-Sectional Prior Fold (b)

Each update of a [`CrossSectionalFactorPrior`](@ref) after its first `partial_fit!` computes the exposures, the regression and the idiosyncratic variance of the new observations alone, from the last panel rows that its descriptors read. Then `prior(pe)` builds the result with the code of the batch fit.

```@docs
prior(pe::CrossSectionalFactorPrior, st::PortfolioOptimisers.CrossSectionalCarryState; strict::Bool = false, kwargs...)
```
