```@meta
Description = "Asset Panel builder, public API of PortfolioOptimisers.jl: AbstractPanelFieldInput, AbstractPanelFillAlgorithm, NumericPanelInput, CategoricalPanelInput, …"
```

# Asset Panel builder

## Types

```@docs
AbstractPanelFieldInput
AbstractPanelFillAlgorithm
NumericPanelInput
CategoricalPanelInput
TensorPanelInput
NoPanelFill
ConstantPanelFill
ForwardPanelFill
BackwardPanelFill
```

## Functions

```@docs
asset_panel
panel_fill
panel_resolve
panel_input_field
panel_input_is_static
```
