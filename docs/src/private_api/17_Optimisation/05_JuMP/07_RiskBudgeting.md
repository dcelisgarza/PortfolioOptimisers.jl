```@meta
Description = "Risk budgeting, private API of PortfolioOptimisers.jl: ProcessedRiskBudgetingAttributes, RiskBudgetingFormulation, RiskBudgetingAlgorithm, …"
```

# Risk budgeting: private API

```@docs
ProcessedRiskBudgetingAttributes
RiskBudgetingFormulation
RiskBudgetingAlgorithm
ProcessedFactorRiskBudgetingAttributes
ProcessedAssetRiskBudgetingAttributes
risk_budgeting_td_defaults
risk_budget_universe_key
_set_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting, w::VecJuMPScalar; strict::Bool = false)
set_risk_budgeting_constraints!
set_rb_mip_w!
```
