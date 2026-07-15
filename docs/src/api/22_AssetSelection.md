# Asset selection

Asset selectors narrow the universe from the data: drop constant columns, keep the best or worst assets by a risk measure, prune redundant ones. They are ordinary returns-preprocessing estimators — they know nothing about pipelines, and a [`Pipeline`](@ref) drives them through [`fit_preprocessing`](@ref) and [`apply_preprocessing`](@ref) like any other step.

The universe a selector chooses on the training window is its **fitted state**. Applying the fitted result to an unseen window replays that universe rather than re-deciding it, which is what makes a selector safe inside cross-validation.

See `docs/adr/0029-asset-selection-is-returns-preprocessing.md` for the design rationale, and [`PortfolioOptimisers.AbstractAssetSelector`](@ref) for the seam every selector shares.

## Scoring assets with a risk measure

A [`ScoreSelector`](@ref) scores each asset by evaluating a risk measure on that asset's own return series, then hands the scores to a rule. Any risk measure whose [`supports_precomputed_returns`](@ref) is `true` may be used, which covers the quantile and drawdown families, the moment measures, and [`MeanReturn`](@ref). [`bigger_is_better`](@ref) tells the ordinal rules which end of the ordering is "best".

Two measures are notable exceptions. [`Variance`](@ref) and [`StandardDeviation`](@ref) are [`WeightsInput`](@ref) measures: their functors consume portfolio weights, not a return series, so they cannot score a single asset and are rejected at construction. Use `SCM()`, which computes the same quantity from a return series — [`ZeroVarianceFilter`](@ref) spells this for you.

```@docs
ScoreSelector
CompleteAssetSelector
```

## Selection rules

A rule turns per-asset scores into a keep-mask. [`ThresholdRule`](@ref) is *literal* — it compares raw scores against absolute bounds and ignores orientation, because a zero-variance filter must drop the *low*-variance assets. [`RankRule`](@ref) and [`QuantileRule`](@ref) are *ordinal* — they consult [`bigger_is_better`](@ref) and take counts (or fractions) from each tail.

Ties at a rank cut are excluded entirely, so an ordinal rule may return fewer assets than asked. If the 20th and 21st assets score equally, `RankRule(; best = 20)` keeps 19: the tied block is dropped rather than split arbitrarily. This is the library's "if we cannot tell them apart, trust neither" tie policy.

```@docs
PortfolioOptimisers.AbstractSelectionRule
ThresholdRule
RankRule
QuantileRule
```

## Discarding redundant assets

A [`RedundancySelector`](@ref) discards assets that duplicate information already carried by others. Its `alg` decides what "redundant" means, and its `score` decides which member of a redundancy group survives.

[`PairwiseCorrelation`](@ref) is greedy: it drops one asset at a time until no surviving pair exceeds the threshold, and never chains. [`CorrelationComponents`](@ref) reads the same correlations transitively, treating a chain `A ~ B ~ C` as one blob and keeping a single representative — a stronger reduction, and a different answer on the same input. [`ClusterGroups`](@ref) partitions with [`clusterise`](@ref) and keeps one representative per cluster.

Leaving `score` as `nothing` falls back to the correlation algorithms' own survivor rule: the asset with the lowest summary correlation to the rest of the universe. [`ClusterGroups`](@ref) has no such fallback and requires a `score`.

```@docs
RedundancySelector
PortfolioOptimisers.AbstractRedundancyAlgorithm
PairwiseCorrelation
CorrelationComponents
ClusterGroups
```

## Internals

```@docs
PortfolioOptimisers.asset_scores
PortfolioOptimisers.rule_keep
PortfolioOptimisers.tail_mask
PortfolioOptimisers.tail_action_mask
PortfolioOptimisers.groups_argbest
PortfolioOptimisers.correlation_components
PortfolioOptimisers.drop_scores
PortfolioOptimisers.redundancy_keep
PortfolioOptimisers.requires_score
PortfolioOptimisers.assert_scoreable
PortfolioOptimisers.assert_selection_action
PortfolioOptimisers.assert_tail_counts
PortfolioOptimisers.assert_correlation_threshold
```
