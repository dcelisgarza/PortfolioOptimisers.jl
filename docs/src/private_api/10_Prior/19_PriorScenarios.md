```@meta
Description = "Prior Scenarios, private API of PortfolioOptimisers.jl: scenario_fill_pairs, scenario_fill_msg, resolve_fill_limit, scenario_window, scenario_ens, …"
```

# [Prior Scenarios: private API](@id private-api-prior-scenarios)

A prior result carries its scenarios in `X`. A cap on the number of scenarios keeps the last rows of the observations. The scenario fill writes zero into each gap of an investable asset, so that a risk measure that reads the scenarios reads a complete matrix. The functions below find the entries to fill, resolve the limit on the filled share, keep the rows under the cap, report the filled entries, and fill them.

```@docs
scenario_fill_pairs
scenario_fill_msg
resolve_fill_limit
PortfolioOptimisers.scenario_window
PortfolioOptimisers.scenario_ens
PortfolioOptimisers.scenario_fill_report
PortfolioOptimisers.scenario_fill_remember!
scenario_fill
```
