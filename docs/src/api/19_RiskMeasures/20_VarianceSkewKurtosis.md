# Variance Skew Kurtosis

```@docs
MaxRiskMeasureSettings
Skewness
bigger_is_better(::Skewness)
factory(r::Skewness, pr::HighOrderPrior, args...; kwargs...)
factory(r::Skewness, pr::LowOrderPrior, args...; kwargs...)
risk_measure_view(r::Skewness{<:Any, <:Any, <:Nothing}, i, args...)
risk_measure_view(r::Skewness{<:Any, <:Any, <:MatNum}, i, args...)
no_risk_expr_risk_measure(r::Skewness)
VarianceSkewKurtosis
factory(r::VarianceSkewKurtosis, pr::AbstractPriorResult, args...; kwargs...)
risk_measure_view(r::VarianceSkewKurtosis, i, args...)
```
