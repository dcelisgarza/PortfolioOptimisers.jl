```@meta
Description = "Asset scorers, public API of PortfolioOptimisers.jl: FoldRiskScorer, AbstractFoldStatistic, Value, Rank, AbstractFoldAggregation, MeanFoldScore, …"
```

# Asset scorers

An asset scorer gives one score to each asset from the whole returns data. [`ScoreSelector`](@ref) and [`RedundancySelector`](@ref) take a scorer in their `score` field, in the place of a risk measure. A risk measure scores each asset on the whole training window. A scorer can score each asset over the folds of a cross-validation scheme, inside the training window of the selector.

## Scores over the test windows

[`FoldRiskScorer`](@ref) evaluates a risk measure on each test window of a cross-validation scheme, and aggregates the values over the windows. One window of bad luck then does not decide the rank of an asset. Each window scores only the assets that it covers, so an asset with a gap in one window keeps its score from the other windows.

The aggregation sets the rule, and its statistic sets what the rule reduces. [`Value`](@ref) keeps the raw value of the measure, so a window of high volatility weighs more in a mean. [`Rank`](@ref) replaces each value with the rank of the asset in its window, so each window weighs the same.

```@docs
FoldRiskScorer
PortfolioOptimisers.AbstractFoldStatistic
Value
Rank
PortfolioOptimisers.AbstractFoldAggregation
MeanFoldScore
MedianFoldScore
WorstFoldScore
```

## Scores from the weights of an optimiser

[`WeightMassScorer`](@ref) runs an optimiser or a pipeline through a cross-validation scheme, and scores each asset by its mean weight or by how often it gets a weight. The score needs the whole cross-section of each fold, so no risk measure on one asset can state it.

```@docs
WeightMassScorer
```

## Scores from the panel fields

[`PanelFieldScorer`](@ref) reads the asset panel of the returns data, not the returns. The value of an asset at each observation is one panel field, a sum or a product of panel fields, or a descriptor such as an [`EWVolumeRatio`](@ref), and a statistic reduces the values over the observations to the score. A price and a volume give the median traded amount, and a [`ThresholdRule`](@ref) on it is a liquidity floor. The orientation of the score is a trait of the statistic, and the `bib` field can override it.

```@docs
PanelFieldScorer
PortfolioOptimisers.bigger_is_better(::PortfolioOptimisers.VectorToScalarMeasure)
```

## Functions

```@docs
PortfolioOptimisers.fold_statistic(::Value, ::PortfolioOptimisers.VecNum, ::Bool)
PortfolioOptimisers.statistic_bigger_is_better
PortfolioOptimisers.fold_aggregate
```
