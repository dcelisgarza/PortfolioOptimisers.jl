# Base Risk Constraints

```@docs
scalarise_risk_expression!
set_risk_constraints!(model::JuMP.Model, r::RiskMeasure, opt::JuMPOptimisationEstimator, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, b1::Option{<:MatNum} = nothing; kwargs...)
risk_frontier_length
set_risk_frontier_owner!
set_risk_upper_bound!
set_risk_expression!
set_risk_bounds_and_expression!
set_range_risk_constraints!
set_drawdown_constraints!
AbstractRiskSeriesAlgorithm
NetReturnsRiskSeries
DrawdownRiskSeries
risk_series
prior_high_order_quantity
assert_high_order_quantity
dup_elim_sum_selector
NonFRCJuMPOpt
```
