```@meta
Description = "Meta optimisation, private API of PortfolioOptimisers.jl: FullUniverse, ClusterUniverse, sub_portfolio_cv, outer_optimisation_finaliser, …"
```

# Meta optimisation: private API

```@docs
FullUniverse
ClusterUniverse
sub_portfolio_cv
outer_optimisation_finaliser
combination_weights
prepare_outer_rd
assert_fold_alignment
fold_row_indices
fold_asset_panel
fold_feature_anchors
panel_field_stack(fs::AbstractVector{<:NumericPanelField})
rebuild_asset_panel
rebuild_returns_result
sub_portfolio_predictions
predict_outer_returns
```
