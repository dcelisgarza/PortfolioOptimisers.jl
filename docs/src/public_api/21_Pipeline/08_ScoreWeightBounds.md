```@meta
Description = "Score weight bounds, public API of PortfolioOptimisers.jl: ScoreWeightBounds."
```

# Score weight bounds

[`ScoreWeightBounds`](@ref) bounds the magnitude of each weight by a multiple of the score of its asset. The score comes from a risk measure or from an [`AbstractAssetScorer`](@ref), such as a [`PanelFieldScorer`](@ref) of the median traded amount. Each fit scores the assets on its own returns, so a fold of a cross-validation reads only its training rows. The estimator goes in the `wb` field of an optimiser, and it tightens the static bounds that its own `wb` field holds.

```@docs
ScoreWeightBounds
```

[`weight_bounds_constraints`](@ref) documents the resolution of the estimator with its other methods.
