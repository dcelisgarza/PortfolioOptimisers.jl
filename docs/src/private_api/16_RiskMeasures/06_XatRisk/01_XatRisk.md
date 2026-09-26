```@meta
Description = "X at Risk, private API of PortfolioOptimisers.jl: ValueatRiskFormulation, CholRM, compute_value_at_risk_z, compute_value_at_risk_cz, …"
```

# X at Risk: private API

```@docs
ValueatRiskFormulation
CholRM
compute_value_at_risk_z
compute_value_at_risk_cz
resolve_deferred_quantities(x::ValueatRisk, pr::AbstractPriorResult)
resolve_deferred_quantities(x::ValueatRiskRange, pr::AbstractPriorResult)
absolute_drawdown_vec
relative_drawdown_vec(x::VecNum)
mip_var_bounds
empirical_value_at_risk
```
