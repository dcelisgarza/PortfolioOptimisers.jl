```@meta
Description = "Prior Scenarios, private API of PortfolioOptimisers.jl: scenario_fill_pairs, scenario_fill_msg, resolve_fill_limit, scenario_window, scenario_weights, …"
```

# [Prior Scenarios: private API](@id private-api-prior-scenarios)

A prior result carries its scenarios in `X`. A cap on the number of scenarios keeps the last rows of the observations. The scenario fill writes zero into each gap of an investable asset, so that a risk measure that reads the scenarios reads a complete matrix. The functions below find the entries to fill, resolve the limit on the filled share, keep the rows under the cap, put the probabilities of the scenarios on the rows that the moments read, report the filled entries, and fill them. A Cross-Sectional Factor Prior reads its scenarios from the factor returns of its regression, so two functions fit that regression and give those rows to an entropy pooling prior around it.

```@docs
scenario_fill_pairs
scenario_fill_msg
resolve_fill_limit
PortfolioOptimisers.scenario_window
PortfolioOptimisers.scenario_weights
PortfolioOptimisers.cross_sectional_regression_stage
PortfolioOptimisers.cross_sectional_factor_scenarios
PortfolioOptimisers.scenario_ens
PortfolioOptimisers.scenario_fill_report
PortfolioOptimisers.scenario_fill_remember!
scenario_fill
```
