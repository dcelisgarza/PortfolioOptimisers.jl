```@meta
Description = "Base cross-sectional factor Prior, public API of PortfolioOptimisers.jl: AbstractSystematicRepair."
```

# Base cross-sectional factor Prior

The Systematic Repair rule of a [`CrossSectionalFactorPrior`](@ref) says which steps of its matrix processing run on the systematic block of the asset covariance before the prior adds the idiosyncratic block. [`NoSystematicRepair`](@ref), the default, and [`SystematicRepair`](@ref) are its two members.

```@docs
AbstractSystematicRepair
```

Every other name of this topic is private. The [private page](@ref private-api-base-cross-sectional-factor-prior) documents them.
