```@meta
Description = "Portfolio and asset fees, public API of PortfolioOptimisers.jl: FeesEstimator, Fees, AmortisedFees, FirstObservationFees, fees_constraints, calc_fees, …"
```

# Portfolio and asset fees

In active, and small quantity investing, fees can be a non-negligible factor that affects portfolio returns. `PortfolioOptimisers.jl` has the capability of including a variety of fees.

```@docs
FeesEstimator
Fees
AmortisedFees
FirstObservationFees
fees_constraints
calc_fees
calc_fixed_fees
calc_asset_fees
calc_asset_fixed_fees
calc_total_fees
calc_total_asset_fees
port_opt_view(fees::Fees, i, X::MatNum, args...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
