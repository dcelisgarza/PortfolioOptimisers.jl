# Portfolio and asset fees

In active, and small quantity investing, fees can be a non-negligible factor that affects portfolio returns. `PortfolioOptimisers.jl` has the capability of including a variety of fees.

```@docs
AbstractFeeAmortisation
FeesEstimator
Fees
AmortisedFees
FirstObservationFees
FeesE_Fees
fees_constraints
calc_fees
calc_fixed_fees
calc_asset_fees
calc_asset_fixed_fees
calc_periodic_fees
calc_one_off_fees
calc_asset_periodic_fees
calc_asset_one_off_fees
calc_liquidation_fees
calc_fixed_liquidation_fees
calc_asset_liquidation_fees
calc_asset_fixed_liquidation_fees
add_liquidation_terms
calc_total_fees
calc_total_asset_fees
override_fee_amortisation
needs_previous_weights(fe::FeesE_Fees)
port_opt_view(fees::Fees, i, X::MatNum, args...)
strip_liquidation_carriers
investable_fees_view
lift_fees
lift_fee_rate
lift_turnover
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
