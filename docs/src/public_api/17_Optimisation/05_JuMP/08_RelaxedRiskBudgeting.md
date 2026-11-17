```@meta
Description = "Relaxed risk budgeting, public API of PortfolioOptimisers.jl: RelaxedRiskBudgetingResult, BasicRelaxedRiskBudgeting, RegularisedRelaxedRiskBudgeting, …"
```

# Relaxed risk budgeting

```@docs
RelaxedRiskBudgetingResult
BasicRelaxedRiskBudgeting
RegularisedRelaxedRiskBudgeting
RegularisedPenalisedRelaxedRiskBudgeting
RelaxedRiskBudgeting
factory(rrb::RelaxedRiskBudgeting, w::AbstractVector)
port_opt_view(rrb::RelaxedRiskBudgeting, i, X::MatNum, args...)
_optimise(rrb::RelaxedRiskBudgeting, rd::ReturnsResult = ReturnsResult())
needs_previous_weights(opt::RelaxedRiskBudgeting)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
