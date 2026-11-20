```@meta
Description = "Risk budgeting, public API of PortfolioOptimisers.jl: RiskBudgetingResult, LogRiskBudgeting, MixedIntegerRiskBudgeting, AssetRiskBudgeting, …"
```

# Risk budgeting

```@docs
RiskBudgetingResult
LogRiskBudgeting
MixedIntegerRiskBudgeting
AssetRiskBudgeting
FactorRiskBudgeting
RiskBudgeting
factory(res::RiskBudgetingResult, fb::Option{<:OptE_Opt_FbChain})
port_opt_view(alg::LogRiskBudgeting{<:VecInt}, i, args...)
factory(rb::RiskBudgeting, w::AbstractVector)
port_opt_view(rb::RiskBudgeting, i, X::MatNum, args...)
_optimise(rb::RiskBudgeting, rd::ReturnsResult = ReturnsResult())
needs_previous_weights(opt::RiskBudgeting)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
