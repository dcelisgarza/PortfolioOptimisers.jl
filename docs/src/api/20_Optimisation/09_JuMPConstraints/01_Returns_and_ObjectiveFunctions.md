# Returns and Objective Functions

```@docs
ArithmeticReturn
LogarithmicReturn
bounds_returns_estimator
no_bounds_returns_estimator(r::ArithmeticReturn, flag::Bool)
no_bounds_optimiser
MinimumRisk
MaximumUtility
MaximumRatio
MaximumReturn
set_maximum_ratio_factor_variables!
set_return_bounds!
set_max_ratio_return_constraints!
add_fees_to_ret!
add_market_impact_cost!
set_return_constraints!
set_ucs_return_constraints!(model::JuMP.Model, ucs::BoxUncertaintySet, mu::Num_VecNum)
set_ucs_return_constraints!(model::JuMP.Model, ucs::EllipsoidalUncertaintySet, mu::Num_VecNum)
set_ucs_return_constraints!(model::JuMP.Model, ucs::L1UncertaintySet, mu::Num_VecNum)
set_ucs_return_constraints!(model::JuMP.Model, ucs::SignedL1UncertaintySet, mu::Num_VecNum)
set_max_ratio_log_return_constraints!
add_to_objective_penalty!
add_penalty_to_objective!
set_portfolio_objective_function!
```
