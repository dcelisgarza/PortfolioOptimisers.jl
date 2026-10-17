```@meta
Description = "Regime switches, public API of PortfolioOptimisers.jl: ExactDebias, LawDebias, RawStatistic, NoHacFloor, PerTermHacFloor, VolatilityBeforeUpdate, …"
```

# Regime switches

## Types

```@docs
ExactDebias
LawDebias
RawStatistic
NoHacFloor
PerTermHacFloor
VolatilityBeforeUpdate
VolatilityAfterUpdate
BiasCorrectedVolatility
RawVolatility
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
