# Returns and Objective Functions

```@docs
JuMPReturnsSettings
VecJRE
JRE_VecJRE
ArithRetMu
ArithmeticReturn
resolve_deferred_quantities(rt::ArithmeticReturn, pr::AbstractPriorResult)
LogarithmicReturn
NoReturn
zero_return_expression_flag
assert_no_return_objective_compatibility
assert_return_term_required
bounds_returns_estimator
no_bounds_returns_estimator(r::ArithmeticReturn, flag::Bool)
no_bounds_returns_settings
unit_scale_returns_estimator
no_bounds_optimiser
MinimumRisk
MaximumUtility
MaximumRatio
MaximumReturn
MaximumElementReturn
set_maximum_ratio_factor_variables!
set_maximum_ratio_normalisation!
set_return_bounds!
set_return_expression!
scalarise_return_expression!
set_max_ratio_return_constraints!
aggregate_return_characteristic
add_fees_to_ret!
add_market_impact_cost!
set_return_constraints!
set_ucs_return_constraints!(model::JuMP.Model, i, ucs::BoxUncertaintySet, mu::Num_VecNum, settings::JuMPReturnsSettings)
set_ucs_return_constraints!(model::JuMP.Model, i, ucs::EllipsoidalUncertaintySet, mu::Num_VecNum, settings::JuMPReturnsSettings)
set_ucs_return_constraints!(model::JuMP.Model, i, ucs::L1UncertaintySet, mu::Num_VecNum, settings::JuMPReturnsSettings)
set_ucs_return_constraints!(model::JuMP.Model, i, ucs::SignedL1UncertaintySet, mu::Num_VecNum, settings::JuMPReturnsSettings)
set_ucs_return_constraints!(model::JuMP.Model, i, ucs::NormBallUncertaintySet{<:Any, <:Any, <:Any, <:MuUncertaintySetClass}, mu::Num_VecNum, settings::JuMPReturnsSettings)
add_to_objective_penalty!
add_penalty_to_objective!
set_portfolio_objective_function!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
