```@meta
Description = "Cross-Sectional Factor Prior, public API of PortfolioOptimisers.jl: CrossSectionalFactorPrior, prior."
```

# Cross-Sectional Factor Prior

A cross-sectional factor prior fits a factor model at each observation from the exposures in an [`AssetPanel`](@ref), and turns the model into moments of the assets. It returns a [`LowOrderPrior`](@ref) whose `rr` field holds the fitted [`CrossSectionalFactorModel`](@ref).

```@docs
CrossSectionalFactorPrior
prior(pe::CrossSectionalFactorPrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, iv::Option{<:MatNum} = nothing, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
prior(pe::CrossSectionalFactorPrior, rd::ReturnsResult; kwargs...)
```
