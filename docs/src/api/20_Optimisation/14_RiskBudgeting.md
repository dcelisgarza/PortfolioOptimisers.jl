# Risk budgeting

```@docs
RiskBudgetingResult
ProcessedFactorRiskBudgetingAttributes
ProcessedAssetRiskBudgetingAttributes
RiskBudgetingFormulation
LogRiskBudgeting
MixedIntegerRiskBudgeting
RiskBudgetingAlgorithm
AssetRiskBudgeting
FactorRiskBudgeting
RiskBudgeting
risk_budgeting_algorithm_view(r::AssetRiskBudgeting, i)
risk_budgeting_algorithm_view(r::FactorRiskBudgeting, i)
set_risk_budgeting_constraints!
set_rb_mip_w!
optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
```
