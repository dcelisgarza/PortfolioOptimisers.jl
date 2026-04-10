# Returns and Objective Functions

```@docs
ArithmeticReturn
LogarithmicReturn
bounds_returns_estimator
no_bounds_returns_estimator(r::ArithmeticReturn, flag::Bool)
MinimumRisk
MaximumUtility
MaximumRatio
MaximumReturn
set_return_bounds!
add_fees_to_ret!
add_market_impact_cost!
set_return_constraints!
set_ucs_return_constraints!(model::JuMP.Model, ucs::BoxUncertaintySet, mu::Num_VecNum)
set_ucs_return_constraints!(model::JuMP.Model, ucs::EllipsoidalUncertaintySet, mu::Num_VecNum)
add_to_objective_penalty!
add_penalty_to_objective!
set_portfolio_objective_function!
```
