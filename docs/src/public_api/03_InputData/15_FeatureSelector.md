```@meta
Description = "Feature selector, public API of PortfolioOptimisers.jl: panel_feature_matrix, panel_field_stack!."
```

# Feature selector

A feature selector names the panel fields that a feature matrix stacks. [`feature_matrix`](@ref),
[`feature_labels`](@ref) and [`LabelGroup`](@ref) are documented with the feature distance that
reads the matrix. This page documents the stack of every field of a panel, and the method that a
new kind of panel field adds to it.

A static panel stacks to `assets × features`. A time-varying panel stacks to `assets × features ×
observations`, with the observations on the last axis, so the matrix of one observation is one
contiguous block.

```@docs
panel_feature_matrix
panel_field_stack!
```

The [private page](@ref private-api-feature-selector) documents the other names of this topic.
