# Portfolio and asset fees

In active, and small quantity investing, fees can be a non-negligible factor that affects portfolio returns. `PortfolioOptimisers.jl` has the capability of including a variety of fees.

```@docs
FeesEstimator
Fees
AmortisedFees
FeesE_Fees
fees_constraints
calc_fees
calc_fixed_fees
calc_asset_fees
calc_asset_fixed_fees
needs_previous_weights(fe::FeesE_Fees)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
