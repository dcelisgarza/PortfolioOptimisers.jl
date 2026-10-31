```@meta
Description = "Base cross-sectional factor Prior, public API of PortfolioOptimisers.jl: AbstractSystematicRepair, AbstractCarryRule, AbstractLeverageOneMask, …"
```

# Base cross-sectional factor Prior

The systematic repair rule of a [`CrossSectionalFactorPrior`](@ref) says which steps of its matrix processing run on the systematic block of the asset covariance before the prior adds the idiosyncratic block. The default `nothing` runs every step except the `:pdm` step, and [`SystematicRepair`](@ref) runs every step.

```@docs
AbstractSystematicRepair
```

The carry rule of a [`CrossSectionalFactorPrior`](@ref) says whether the carry fold accepts a part whose step cost grows with the stream. [`FoldOrRefit`](@ref), the default, and [`FoldOnly`](@ref) are its two members.

```@docs
AbstractCarryRule
```

The leverage-one rule of a [`CrossSectionalFactorPrior`](@ref) says how the prior reads the residual of an asset that is the only member of its level at an observation, when it scales the idiosyncratic returns. [`MaskedLeverageOne`](@ref), the default, reads it as a gap, and [`UnmaskedLeverageOne`](@ref) reads it as it stands.

```@docs
AbstractLeverageOneMask
MaskedLeverageOne
UnmaskedLeverageOne
PortfolioOptimisers.leverage_one_residuals
```

Every other name of this topic is private. The [private page](@ref private-api-base-cross-sectional-factor-prior) documents them.
