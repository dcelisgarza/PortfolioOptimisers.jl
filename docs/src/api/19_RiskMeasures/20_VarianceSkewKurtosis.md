# Variance Skew Kurtosis

```@docs
MaxRiskMeasureSettings
Skewness
bigger_is_better(::Skewness)
factory(r::Skewness, pr::HighOrderPrior, args...; kwargs...)
factory(r::Skewness, pr::LowOrderPrior, args...; kwargs...)
port_opt_view(r::Skewness, i, args...)
port_opt_view(r::Skewness{<:Any, <:Any, <:MatNum}, i, args...)
no_risk_expr_risk_measure(r::Skewness)
no_bounds_no_risk_expr_risk_measure(r::Skewness)
bounds_risk_measure(r::Skewness, ub::Number)
VarianceSkewKurtosis
resolve_deferred_quantities(r::VarianceSkewKurtosis, pr::AbstractPriorResult)
factory(r::VarianceSkewKurtosis, pr::AbstractPriorResult, args...; kwargs...)
supports_precomputed_returns(r::Skewness)
supports_precomputed_returns(::VarianceSkewKurtosis)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
