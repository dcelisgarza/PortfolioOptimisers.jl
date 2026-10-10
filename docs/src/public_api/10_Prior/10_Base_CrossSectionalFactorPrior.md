```@meta
Description = "Base cross-sectional factor Prior, public API of PortfolioOptimisers.jl: AbstractSystematicRepair, AbstractCarryRule."
```

# Base cross-sectional factor Prior

The systematic repair rule of a [`CrossSectionalFactorPrior`](@ref) says which steps of its matrix processing run on the systematic block of the asset covariance before the prior adds the idiosyncratic block. [`NoSystematicRepair`](@ref), the default, and [`SystematicRepair`](@ref) are its two members.

```@docs
AbstractSystematicRepair
```

The carry rule of a [`CrossSectionalFactorPrior`](@ref) says whether the carry fold accepts a part whose step cost grows with the stream. [`FoldOrRefit`](@ref), the default, and [`FoldOnly`](@ref) are its two members.

```@docs
AbstractCarryRule
```

Every other name of this topic is private. The [private page](@ref private-api-base-cross-sectional-factor-prior) documents them.
