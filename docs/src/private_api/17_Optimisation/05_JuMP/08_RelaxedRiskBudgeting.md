```@meta
Description = "Relaxed risk budgeting, private API of PortfolioOptimisers.jl: RelaxedRiskBudgetingAlgorithm, relaxed_risk_budgeting_td_defaults, …"
```

# Relaxed risk budgeting: private API

```@docs
RelaxedRiskBudgetingAlgorithm
relaxed_risk_budgeting_td_defaults
set_relaxed_risk_budgeting_alg_constraints!
_set_relaxed_risk_budgeting_constraints!(model::JuMP.Model, rrb::RelaxedRiskBudgeting, x::VecJuMPScalar, A::MatNum, w::VecJuMPScalar, sigma::MatNum, chol::Option{<:MatNum}, z::VecJuMPScalar, sigma_z::MatNum)
set_relaxed_risk_budgeting_constraints!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
