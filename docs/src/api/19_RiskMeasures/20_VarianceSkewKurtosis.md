# Variance Skew Kurtosis

```@docs
MaxRiskMeasureSettings
Skewness
bigger_is_better(::Skewness)
factory(r::Skewness, pr::HighOrderPrior, args...; kwargs...)
factory(r::Skewness, pr::LowOrderPrior, args...; kwargs...)
port_opt_view(r::Skewness{<:Any, <:Any, <:Nothing}, i, args...)
port_opt_view(r::Skewness{<:Any, <:Any, <:MatNum}, i, args...)
no_risk_expr_risk_measure(r::Skewness)
no_bounds_no_risk_expr_risk_measure(r::Skewness)
bounds_risk_measure(r::Skewness, ub::Number)
VarianceSkewKurtosis
factory(r::VarianceSkewKurtosis, pr::AbstractPriorResult, args...; kwargs...)
port_opt_view(r::VarianceSkewKurtosis, i, args...)
supports_precomputed_returns(r::Skewness)
supports_precomputed_returns(::VarianceSkewKurtosis)
```
