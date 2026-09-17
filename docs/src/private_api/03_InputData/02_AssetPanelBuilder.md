```@meta
Description = "Asset Panel builder, private API of PortfolioOptimisers.jl: AbstractAssetPanelEstimator, AbstractPanelFieldInput, AbstractPanelFillAlgorithm, …"
```

# Asset Panel builder: private API

## Types

```@docs
PortfolioOptimisers.AbstractAssetPanelEstimator
PortfolioOptimisers.AbstractPanelFieldInput
PortfolioOptimisers.AbstractPanelFillAlgorithm
```

## Functions

```@docs
PortfolioOptimisers.panel_build_observations
PortfolioOptimisers.panel_fill
PortfolioOptimisers.panel_fill_array
PortfolioOptimisers.panel_directional_fill
PortfolioOptimisers.panel_resolve
PortfolioOptimisers.panel_input_field
PortfolioOptimisers.panel_input_is_static
PortfolioOptimisers.is_panel_blank
PortfolioOptimisers.assert_panel_fill
PortfolioOptimisers.assert_panel_input
PortfolioOptimisers.assert_categorical_fill
PortfolioOptimisers.assert_panel_input_fill
```
