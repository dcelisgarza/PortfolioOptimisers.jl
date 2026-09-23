```@meta
Description = "Asset selection, public API of PortfolioOptimisers.jl: ScoreSelector, CompleteAssetSelector, AbstractSelectionRule, ThresholdRule, RankRule, QuantileRule, …"
```

# Asset selection

An asset selector removes assets from the universe, from the data. It can drop constant columns, keep the best or worst assets by a risk measure, or remove redundant assets. A selector is an ordinary preprocessing estimator of returns, and it does not depend on a pipeline. A [`Pipeline`](@ref) calls [`fit_preprocessing`](@ref) and [`apply_preprocessing`](@ref) on it, as on any other step.

A selector chooses its assets on the training window, and that choice is its fitted result. Applied to a later window, the fitted result keeps the same assets and does not choose again. The test window never changes the choice, so you can use a selector inside cross-validation.

Every selector subtypes [`PortfolioOptimisers.AbstractAssetSelector`](@ref).

## Scoring assets with a risk measure

A [`ScoreSelector`](@ref) computes a risk measure on the returns of each asset alone, and passes the scores to a rule. It accepts any risk measure whose [`supports_precomputed_returns`](@ref) is `true`, which includes the quantile and drawdown measures, the moment measures and [`MeanReturn`](@ref). [`bigger_is_better`](@ref) tells the ordinal rules which end of the order is the best.

[`Variance`](@ref) and [`StandardDeviation`](@ref) do not work here. They are [`WeightsInput`](@ref) measures, which take portfolio weights and not a series of returns, so they cannot score one asset, and the constructor throws an error for them. Use `SCM()`, which computes the same quantity from a series of returns. [`ZeroVarianceFilter`](@ref) builds that selector for you.

```@docs
ScoreSelector
CompleteAssetSelector
```

## Selection rules

A rule turns the scores into a mask of the assets to keep. [`ThresholdRule`](@ref) compares the raw scores with fixed bounds, and ignores which end is better, because a zero-variance filter must drop the assets with low variance. [`RankRule`](@ref) and [`QuantileRule`](@ref) read [`bigger_is_better`](@ref), and take a count or a fraction of the assets from each end.

An ordinal rule drops every asset of a tie at its cut, so it can keep fewer assets than you ask for. If the 20th and 21st assets have the same score, `RankRule(; best = 20)` keeps 19. The rule drops both tied assets, because it has no reason to keep one of them and not the other.

```@docs
PortfolioOptimisers.AbstractSelectionRule
ThresholdRule
RankRule
QuantileRule
```

## Discarding redundant assets

A [`RedundancySelector`](@ref) removes assets whose returns carry the same information as the returns of other assets. Its `alg` field decides what counts as redundant, and its `score` field decides which asset of a redundant group stays.

[`PairwiseCorrelation`](@ref) removes one asset at a time until no remaining pair has a correlation above the threshold, and it does not follow chains. [`CorrelationComponents`](@ref) follows chains of the same correlations. If `A` is close to `B` and `B` is close to `C`, it puts `A`, `B` and `C` in one group and keeps one of them. It removes more assets than `PairwiseCorrelation`, and on the same data it can keep a different set. [`ClusterGroups`](@ref) groups the assets with [`clusterise`](@ref) and keeps one asset per cluster.

If you leave `score` as `nothing`, the two correlation algorithms keep the asset with the lowest summary correlation to the rest of the universe. [`ClusterGroups`](@ref) has no such default, and it needs a `score`.

[`ClusterGroups`](@ref) is also the only redundancy algorithm that takes a distance estimator, and so the only one that can group assets by a feature matrix in place of their returns. The other two take a `StatsBase.CovarianceEstimator`. Give its `cle` field a [`FeatureDistance`](@ref), and the groups come from data outside the returns. For example, a sector classification, held as a categorical field of the asset panel through [`panel_input`](@ref), leaves one asset per sector, not one per group of correlated assets. The selector reads the panel directly from the [`ReturnsResult`](@ref), because asset selection runs before a prior exists. An asset panel estimator that reads a prior throws an error here, and [`PhylogenyPanel`](@ref), which reads none, works.

```@docs
PortfolioOptimisers.AbstractRedundancyAlgorithm
RedundancySelector
PairwiseCorrelation
CorrelationComponents
ClusterGroups
```

## Functions

```@docs
PortfolioOptimisers.asset_scores
PortfolioOptimisers.rule_keep
PortfolioOptimisers.redundancy_keep
PortfolioOptimisers.requires_score
```
