# Base JuMP Optimisation

```@docs
BaseJuMPOptimisationEstimator
JuMPOptimisationEstimator
RiskJuMPOptimisationEstimator
ObjectiveFunction
JuMPReturnsEstimator
factory(r::JuMPReturnsEstimator, args...; kwargs...)
jump_returns_view
JuMPConstraintEstimator
CustomJuMPConstraint
CustomJuMPObjective
needs_previous_weights(::CustomJuMPConstraint)
needs_previous_weights(::CustomJuMPObjective)
custom_constraint_view
custom_objective_view
JuMPOptimisationSolution
add_custom_objective_term!
add_custom_constraint!
process_model
optimise_JuMP_model!
set_model_scales!
set_initial_w!
set_w!
set_portfolio_returns!
set_net_portfolio_returns!
set_portfolio_returns_plus_one!
set_portfolio_drawdowns_plus_one!
set_risk_constraints!
```
