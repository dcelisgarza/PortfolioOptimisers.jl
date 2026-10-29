```@meta
Description = "Prior Scenarios, private API of PortfolioOptimisers.jl: scenario_fill_pairs, scenario_fill_msg, resolve_fill_limit, scenario_window, scenario_tail, …"
```

# [Prior Scenarios: private API](@id private-api-prior-scenarios)

A prior result carries its scenarios in `X`. A cap on the number of scenarios keeps the last rows of the observations. The scenario fill writes zero into each gap of an investable asset, so that a risk measure that reads the scenarios reads a complete matrix. The functions below find the entries to fill, resolve the limit on the filled share, keep the rows under the cap, cut a nested prior result to the rows that a wrapping prior keeps, put the probabilities of the scenarios on the rows that the moments read, report the filled entries, and fill them. A Cross-Sectional Factor Prior reads its scenarios from the factor returns of its regression, so two functions fit that regression and give those rows to an entropy pooling prior around it.
A fit with `cache_reg = true` stores a cache, and six functions store it or read it, so the refit of an entropy pooling prior runs no regression. They store the cache on the block, find the block of the cache, and read its factor returns. They split the return forecast of the block, and run the steps of the assembly that read the factor moments. The last one keeps the cache of a pool in the factor slot. Three more functions count the scenarios of a prior with only the steps that drop rows, and check the rows that the exposure lag leaves.

```@docs
scenario_fill_pairs
scenario_fill_msg
resolve_fill_limit
PortfolioOptimisers.scenario_window
PortfolioOptimisers.scenario_tail
PortfolioOptimisers.scenario_weights
PortfolioOptimisers.cross_sectional_regression_stage
PortfolioOptimisers.cross_sectional_factor_scenarios
PortfolioOptimisers.cross_sectional_cached_block
PortfolioOptimisers.cross_sectional_cached_scenarios
PortfolioOptimisers.cross_sectional_forecast_split
PortfolioOptimisers.cross_sectional_moment_prior
PortfolioOptimisers.cross_sectional_kept_cache
PortfolioOptimisers.cross_sectional_stored_cache
PortfolioOptimisers.kept_scenarios
PortfolioOptimisers.scenario_count
PortfolioOptimisers.assert_cross_sectional_lag_rows
PortfolioOptimisers.scenario_moment_counts
PortfolioOptimisers.scenario_fill_report
PortfolioOptimisers.scenario_fill_remember!
scenario_fill
```
