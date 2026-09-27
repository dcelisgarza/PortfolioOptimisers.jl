```@meta
Description = "Portfolio and asset fees, public API of PortfolioOptimisers.jl: FeesEstimator, Fees, AmortisedFees, FirstObservationFees, fees_constraints, calc_fees, …"
```

# Portfolio and asset fees

Fees lower the return of a portfolio, and they can be large for an active strategy or a small portfolio. The types below state the fees that an optimisation deducts, such as proportional and fixed fees on long and short positions, and a fee on the turnover. `AmortisedFees` spreads the fixed fees evenly over the holding period, and `FirstObservationFees` charges them once, on the first observation.

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
needs_previous_weights(fe::FeesE_Fees)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
