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

```@docs
PortfolioOptimisers.AbstractCrossSectionalTransform
cross_sectional_transform
cross_sectional_groups
CrossSectionalWinsoriser
CrossSectionalTanhShrinker
CrossSectionalStandardiser
CrossSectionalGaussianRank
CrossSectionalPercentileRank
PortfolioOptimisers.CS_MISSING_GROUP
PortfolioOptimisers.CS_MAD_CONSISTENCY
PortfolioOptimisers.assert_cross_sectional_matrix
PortfolioOptimisers.assert_cross_sectional_weights
PortfolioOptimisers.assert_cross_sectional_groups
PortfolioOptimisers.cross_sectional_groups_observed!
PortfolioOptimisers.cross_sectional_estimation_mask
PortfolioOptimisers.cross_sectional_indices
PortfolioOptimisers.cross_sectional_weight_type
PortfolioOptimisers.cross_sectional_weighted_mean
PortfolioOptimisers.cross_sectional_equal_std
PortfolioOptimisers.cross_sectional_stat
PortfolioOptimisers.cross_sectional_blank_row!
PortfolioOptimisers.cross_sectional_zscore_row!
PortfolioOptimisers.cross_sectional_recentre_rescale!
PortfolioOptimisers.cross_sectional_midranks!
PortfolioOptimisers.cross_sectional_rank_counts
PortfolioOptimisers.cross_sectional_row_groups
PortfolioOptimisers.cross_sectional_group_split
PortfolioOptimisers.cross_sectional_percentile_ranks
PortfolioOptimisers.cross_sectional_standardise!
PortfolioOptimisers.cross_sectional_cell_stats
```
