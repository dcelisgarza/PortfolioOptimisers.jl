```@meta
Description = "Asset scorers, private API of PortfolioOptimisers.jl: distinct_test_windows, window_statistic, min_folds_scores, weight_mass_grid, weight_mass_candidates, …"
```

# Asset scorers: private API

## The test windows of a fold risk scorer

A [`FoldRiskScorer`](@ref) reads each distinct test window once. In each window it scores only the assets that the window covers, the assets whose return is finite and active at every row of the window. An asset that too few windows cover has no score, and its entry is `NaN`.

```@docs
PortfolioOptimisers.distinct_test_windows
PortfolioOptimisers.window_statistic
PortfolioOptimisers.min_folds_scores
```

## The splits of a weight mass scorer

A [`WeightMassScorer`](@ref) builds one candidate for each point of its grid, finds the prediction of each split once, and maps the weights of each split to the assets of the returns data by name.

```@docs
PortfolioOptimisers.weight_mass_grid
PortfolioOptimisers.weight_mass_candidates
PortfolioOptimisers.split_assets
PortfolioOptimisers.split_offered
PortfolioOptimisers.prediction_at
PortfolioOptimisers.split_prediction
PortfolioOptimisers.split_weight_record
PortfolioOptimisers.weight_mass_value
```
