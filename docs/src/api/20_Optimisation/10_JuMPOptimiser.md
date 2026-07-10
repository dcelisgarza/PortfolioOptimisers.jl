# JuMP Optimiser

```@docs
ProcessedAttributes
ProcessedJuMPOptimiserAttributes
JuMPOptimisationResult
assert_finite_nonnegative_real_or_vec
JuMPOptimiser
needs_previous_weights(opt::JuMPOptimiser)
is_time_dependent(opt::JuMPOptimiser)
update_time_dependent_estimator(opt::JuMPOptimiser, ctx::TimeDependentContext)
factory(opt::JuMPOptimiser, w::AbstractVector)
port_opt_view(opt::JuMPOptimiser, i, X::MatNum, args...)
processed_jump_optimiser_attributes
processed_jump_optimiser
assemble_jump_model!
set_risk_and_scalarise!
jump_optimiser_from_attributes
```
