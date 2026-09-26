```@meta
Description = "Return constraints, private API of PortfolioOptimisers.jl: set_return_bounds!, set_return_expression!, scalarise_return_expression!, …"
```

# [Return constraints: private API](@id private-api-return-constraints)

```@docs
set_return_bounds!
set_return_expression!
scalarise_return_expression!
set_max_ratio_return_constraints!
aggregate_return_characteristic
add_fees_to_ret!
add_market_impact_cost!
set_return_constraints!
set_ucs_return_constraints!(model::JuMP.Model, i, ucs::BoxUncertaintySet, mu::Num_VecNum, settings::JuMPReturnsSettings)
set_ucs_return_constraints!(model::JuMP.Model, i, ucs::EllipsoidalUncertaintySet, mu::Num_VecNum, settings::JuMPReturnsSettings)
set_ucs_return_constraints!(model::JuMP.Model, i, ucs::L1UncertaintySet, mu::Num_VecNum, settings::JuMPReturnsSettings)
set_ucs_return_constraints!(model::JuMP.Model, i, ucs::SignedL1UncertaintySet, mu::Num_VecNum, settings::JuMPReturnsSettings)
set_ucs_return_constraints!(model::JuMP.Model, i, ucs::NormBallUncertaintySet{<:Any, <:Any, <:Any, <:MuUncertaintySetClass}, mu::Num_VecNum, settings::JuMPReturnsSettings)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
