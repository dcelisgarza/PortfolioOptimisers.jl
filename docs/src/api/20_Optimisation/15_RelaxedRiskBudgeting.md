# Relaxed risk budgeting

```@docs
RelaxedRiskBudgetingAlgorithm
BasicRelaxedRiskBudgeting
RegularisedRelaxedRiskBudgeting
RegularisedPenalisedRelaxedRiskBudgeting
RelaxedRiskBudgeting
needs_previous_weights(opt::RelaxedRiskBudgeting)
factory(rrb::RelaxedRiskBudgeting, w::AbstractVector)
opt_view(rrb::RelaxedRiskBudgeting, i, X::MatNum)
set_relaxed_risk_budgeting_alg_constraints!
_set_relaxed_risk_budgeting_constraints!(model::JuMP.Model, rrb::RelaxedRiskBudgeting, w::VecJuMPScalar, sigma::MatNum, chol::Option{<:MatNum})
set_relaxed_risk_budgeting_constraints!
optimise(rrb::RelaxedRiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
```
