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
Base.getproperty(r::RelaxedRiskBudgetingResult, sym::Symbol)
factory(rrb::RelaxedRiskBudgeting, w::AbstractVector)
port_opt_view(rrb::RelaxedRiskBudgeting, i, X::MatNum, args...)
optimise(rrb::RelaxedRiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
