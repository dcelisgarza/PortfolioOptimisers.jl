```@meta
Description = "Prediction weights, public API of PortfolioOptimisers.jl: prediction_weights."
```

# Prediction weights

[`prediction_weights`](@ref) gives the weights of a prediction on asset names, with one row for each portfolio that the prediction holds: one fold, each fold of a path, or each path of a population. [`portfolio_report`](@ref) writes its weights tables from these rows.

```@docs
prediction_weights
```
