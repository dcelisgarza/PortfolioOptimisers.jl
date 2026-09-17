```@meta
Description = "Cross-Sectional Factor Prior, public API of PortfolioOptimisers.jl: CrossSectionalFactorPrior, prior."
```

# Cross-Sectional Factor Prior

A cross-sectional factor prior estimates a point-in-time factor model from an [`AssetPanel`](@ref) and lifts it onto the assets, returning a [`CrossSectionalFactorModel`](@ref) in the `rr` slot of its [`LowOrderPrior`](@ref).

```@docs
CrossSectionalFactorPrior
prior(pe::CrossSectionalFactorPrior, X::MatNum, F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, iv::Option{<:MatNum} = nothing, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
prior(pe::CrossSectionalFactorPrior, rd::ReturnsResult; kwargs...)
```
