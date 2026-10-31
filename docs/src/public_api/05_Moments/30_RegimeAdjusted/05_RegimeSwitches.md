```@meta
Description = "Regime switches, public API of PortfolioOptimisers.jl: ExactDebias, LawDebias, PerTermHacFloor, VolatilityBeforeUpdate, VolatilityAfterUpdate, …"
```

# Regime switches

## Types

```@docs
ExactDebias
LawDebias
PerTermHacFloor
VolatilityBeforeUpdate
VolatilityAfterUpdate
BiasCorrectedVolatility
RefuseIndefiniteWarmUp
ClipIndefiniteBlock
ScoreIndefiniteBlock
```

## Functions

```@docs
debiases
reads_dependence
hac_floor!
volatility_before_update
correlation_variance
indefinite_block
```
