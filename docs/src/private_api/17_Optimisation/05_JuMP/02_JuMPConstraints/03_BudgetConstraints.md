```@meta
Description = "Budget Constraints, private API of PortfolioOptimisers.jl: BudgetConstraintEstimator, Num_BgtCE, BudgetEstimator, BudgetCostEstimator, Num_BgtRg, …"
```

# Budget Constraints: private API

```@docs
BudgetConstraintEstimator
Num_BgtCE
BudgetEstimator
BudgetCostEstimator
Num_BgtRg
set_budget_constraints!
set_long_short_budget_constraints!
set_cost_budget_constraints!
set_gross_budget_constraints!(model::JuMP.Model, gbgt::Number)
set_gross_budget_constraints!(model::JuMP.Model, gbgt::BudgetRange)
assert_gross_budget_admissible
set_exact_budget_constraints!
```
