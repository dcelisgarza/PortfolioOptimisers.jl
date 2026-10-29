```@meta
Description = "Elastic net, private API of PortfolioOptimisers.jl: ElasticNetFit, StatsAPI.coef, StatsAPI.dof, residual_count, elastic_net_design, elastic_net_sweeps, …"
```

# Elastic net: private API

These are the fitted model of [`ElasticNet`](@ref), its counts and scores, and the steps of its fit. `elastic_net_sweeps` runs the coordinate descent, `elastic_net_polish` solves the optimum on a sign pattern exactly, and `elastic_net_counts` reads the effective degrees of freedom off the equations of the active set.

```@docs
PortfolioOptimisers.ElasticNetFit
PortfolioOptimisers.StatsAPI.coef(m::PortfolioOptimisers.ElasticNetFit)
PortfolioOptimisers.StatsAPI.dof(m::PortfolioOptimisers.ElasticNetFit)
PortfolioOptimisers.residual_count(m::PortfolioOptimisers.ElasticNetFit, ::MatNum)
PortfolioOptimisers.elastic_net_design
PortfolioOptimisers.elastic_net_sweeps
PortfolioOptimisers.elastic_net_polish
PortfolioOptimisers.elastic_net_counts
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
