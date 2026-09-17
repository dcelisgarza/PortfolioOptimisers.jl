```@meta
Description = "Cross-sectional transforms, public API of PortfolioOptimisers.jl: AbstractCrossSectionalTransform, CrossSectionalWinsoriser, CrossSectionalTanhShrinker, …"
```

# Cross-sectional transforms

## Cross-sectional transforms

A **cross-sectional transform** rescales one observation of an `observations × assets` matrix
against the other assets of that same observation. No member reads a second observation and no
member is fitted, so a transform is configuration alone and it runs on a plain matrix.

The **estimation set** of an observation is what its statistics are computed from: the finite cells
carrying a positive benchmark weight when `w` is given, and the finite cells otherwise. A cell
outside that set is still transformed against it, so an asset the benchmark does not hold is scored
on the same scale as one it does. An observation whose estimation set is empty returns a `NaN` at
every asset.

The benchmark weights and the group labels are **arguments** of
[`cross_sectional_transform`](@ref), never fields, because one transform runs against a different
benchmark and a different classification at every call site.
[`cross_sectional_groups`](@ref) reads the labels off the codes of a
[`CategoricalPanelField`](@ref).

## Types

```@docs
AbstractCrossSectionalTransform
CrossSectionalWinsoriser
CrossSectionalTanhShrinker
CrossSectionalStandardiser
CrossSectionalGaussianRank
CrossSectionalPercentileRank
```

## Functions

```@docs
cross_sectional_transform
cross_sectional_groups
```
