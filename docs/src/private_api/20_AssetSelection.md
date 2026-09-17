```@meta
Description = "Asset selection, private API of PortfolioOptimisers.jl: AbstractSelectionRule, AbstractRedundancyAlgorithm, tail_mask, tail_action_mask, groups_argbest, …"
```

# Asset selection: private API

## Selection rules

A rule turns per-asset scores into a keep-mask. [`ThresholdRule`](@ref) is *literal* — it compares raw scores against absolute bounds and ignores orientation, because a zero-variance filter must drop the *low*-variance assets. [`RankRule`](@ref) and [`QuantileRule`](@ref) are *ordinal* — they consult [`bigger_is_better`](@ref) and take counts (or fractions) from each tail.

Ties at a rank cut are excluded entirely, so an ordinal rule may return fewer assets than asked. If the 20th and 21st assets score equally, `RankRule(; best = 20)` keeps 19: the tied block is dropped rather than split arbitrarily. This is the library's "if we cannot tell them apart, trust neither" tie policy.

```@docs
PortfolioOptimisers.AbstractSelectionRule
```

## Discarding redundant assets

A [`RedundancySelector`](@ref) discards assets that duplicate information already carried by others. Its `alg` decides what "redundant" means, and its `score` decides which member of a redundancy group survives.

[`PairwiseCorrelation`](@ref) is greedy: it drops one asset at a time until no surviving pair exceeds the threshold, and never chains. [`CorrelationComponents`](@ref) reads the same correlations transitively, treating a chain `A ~ B ~ C` as one blob and keeping a single representative — a stronger reduction, and a different answer on the same input. [`ClusterGroups`](@ref) partitions with [`clusterise`](@ref) and keeps one representative per cluster.

Leaving `score` as `nothing` falls back to the correlation algorithms' own survivor rule: the asset with the lowest summary correlation to the rest of the universe. [`ClusterGroups`](@ref) has no such fallback and requires a `score`.

[`ClusterGroups`](@ref) is also the only redundancy algorithm that reaches a distance estimator — the other two carry a `StatsBase.CovarianceEstimator` — so it is the only one that can be driven by a feature matrix rather than by the returns. Give its `cle` a [`FeatureDistance`](@ref) and the redundancy groups come from exogenous structure: a sector taxonomy, carried as a categorical Panel Field through [`panel_input`](@ref), reduces the universe to one representative per classification, not per correlated blob. The panel is read straight off the [`ReturnsResult`](@ref), because preselection runs before any prior exists — a producer that reads a prior raises here, and [`PhylogenyPanel`](@ref) is the one that does not.

```@docs
PortfolioOptimisers.AbstractRedundancyAlgorithm
```

## Functions

```@docs
PortfolioOptimisers.tail_mask
PortfolioOptimisers.tail_action_mask
PortfolioOptimisers.groups_argbest
PortfolioOptimisers.correlation_components
PortfolioOptimisers.drop_scores
PortfolioOptimisers.assert_scoreable
PortfolioOptimisers.assert_selection_action
PortfolioOptimisers.assert_tail_counts
PortfolioOptimisers.assert_correlation_threshold
```
