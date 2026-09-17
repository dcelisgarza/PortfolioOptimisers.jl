```@meta
Description = "Preprocessing, public API of PortfolioOptimisers.jl: fit_preprocessing, apply_preprocessing."
```

# Preprocessing

## Preprocessing estimators

Preprocessing estimators transform price- or returns-level data under a **fit/apply contract**: [`fit_preprocessing`](@ref) learns whatever state the transformation needs from a training window — the surviving asset universe, imputation parameters, thresholds — and [`apply_preprocessing`](@ref) replays that state on unseen windows, so no information flows from test data back into the transformation.

They are ordinary estimators and know nothing about pipelines. A [`Pipeline`](@ref) drives them through these two verbs, exactly as it drives prior estimators through [`prior`](@ref) or optimisers through [`optimise`](@ref).

```@docs
fit_preprocessing
apply_preprocessing
```
