```@meta
Description = "Variance Skew Kurtosis, public API of PortfolioOptimisers.jl: MaxRiskMeasureSettings, Skewness, VarianceSkewKurtosis, factory, port_opt_view, …"
```

# Variance Skew Kurtosis

```@docs
MaxRiskMeasureSettings
Skewness
VarianceSkewKurtosis
factory(r::Skewness, pr::HighOrderPrior, args...; kwargs...)
factory(r::Skewness, pr::LowOrderPrior, args...; kwargs...)
port_opt_view(r::Skewness, i, args...)
port_opt_view(r::Skewness{<:Any, <:Any, <:MatNum}, i, args...)
no_risk_expr_risk_measure(r::Skewness)
no_bounds_no_risk_expr_risk_measure(r::Skewness)
bounds_risk_measure(r::Skewness, ub::Number)
factory(r::VarianceSkewKurtosis, pr::AbstractPriorResult, args...; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
