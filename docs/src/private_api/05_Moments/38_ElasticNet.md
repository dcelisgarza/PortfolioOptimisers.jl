```@meta
Description = "Elastic net, private API of PortfolioOptimisers.jl: ElasticNetFit, StatsAPI.coef, StatsAPI.predict, elastic_net_design, elastic_net_sweeps, …"
```

# Elastic net: private API

These are the fitted model of [`ElasticNet`](@ref) and the steps of its fit. `elastic_net_sweeps` runs the coordinate descent, and `elastic_net_polish` solves the optimum on a sign pattern exactly.

```@docs
PortfolioOptimisers.ElasticNetFit
PortfolioOptimisers.StatsAPI.coef(m::PortfolioOptimisers.ElasticNetFit)
PortfolioOptimisers.StatsAPI.predict(m::PortfolioOptimisers.ElasticNetFit, X::MatNum)
PortfolioOptimisers.elastic_net_design
PortfolioOptimisers.elastic_net_sweeps
PortfolioOptimisers.elastic_net_polish
```
