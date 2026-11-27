```@meta
Description = "Base meta optimisation, private API of PortfolioOptimisers.jl: FullUniverse, ClusterUniverse, sub_portfolio_cv, enough_solved, dropped_sub_portfolios, …"
```

# Base meta optimisation: private API

```@docs
FullUniverse
ClusterUniverse
sub_portfolio_cv
enough_solved
dropped_sub_portfolios
warn_dropped_sub_portfolios
failed_outer_result
outer_optimisation_finaliser
combination_weights
prepare_outer_rd
outer_prior_rows
assert_fold_alignment
fold_row_indices
fold_asset_panel
fold_feature_anchors
panel_field_stack(fs::AbstractVector{<:NumericPanelField})
rebuild_asset_panel
rebuild_fold_rates
rebuild_returns_result
sub_portfolio_predictions
outer_returns_source
predict_outer_returns
assert_meta_scorer
```
