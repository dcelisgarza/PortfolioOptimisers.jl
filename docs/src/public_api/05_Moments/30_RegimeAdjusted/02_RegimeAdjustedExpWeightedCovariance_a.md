```@meta
Description = "Regime adjusted exp weighted covariance (a), public API of PortfolioOptimisers.jl: RegimeAdjustedTarget, MahalanobisTarget, DiagonalTarget, PortfolioTarget, …"
```

# Regime adjusted exp weighted covariance (a)

## Types

```@docs
RegimeAdjustedTarget
MahalanobisTarget
DiagonalTarget
PortfolioTarget
RegimeAdjustedExpWeightedCovariance
```

## Functions

```@docs
min_active_assets
regime_statistic
regime_kappa
regime_denom
get_regime_state(::RootMeanSquaredAdjusted, ::RegimeAdjustedTarget, stats::VecNum, n::Integer, ::Any)
get_regime_state(method::FirstMomentRegimeAdjusted, target::RegimeAdjustedTarget, stats::VecNum, n::Integer, ::Any)
get_regime_state(method::LogRegimeAdjusted, target::RegimeAdjustedTarget, stats::VecNum, n::Integer, min_val::Number)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
