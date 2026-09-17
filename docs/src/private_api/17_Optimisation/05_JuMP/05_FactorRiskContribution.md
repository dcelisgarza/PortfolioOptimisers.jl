```@meta
Description = "Factor risk contribution, private API of PortfolioOptimisers.jl: factor_risk_contribution_td_defaults, needs_previous_weights, …"
```

# Factor risk contribution: private API

```@docs
factor_risk_contribution_td_defaults
needs_previous_weights(opt::FactorRiskContribution)
set_factor_risk_contribution_constraints!(model::JuMP.Model, re::RegE_Reg, rd::ReturnsResult, pr::Option{<:AbstractPriorResult}, flag::Bool, wi::Option{<:VecNum})
```
