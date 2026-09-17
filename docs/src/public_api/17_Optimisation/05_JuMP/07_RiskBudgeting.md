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
Base.getproperty(r::RiskBudgetingResult, sym::Symbol)
port_opt_view(::RiskBudgetingFormulation, ::Any, args...)
port_opt_view(alg::LogRiskBudgeting{Nothing}, i, args...)
port_opt_view(alg::LogRiskBudgeting{<:VecInt}, i, args...)
factory(rb::RiskBudgeting, w::AbstractVector)
port_opt_view(rb::RiskBudgeting, i, X::MatNum, args...)
optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
