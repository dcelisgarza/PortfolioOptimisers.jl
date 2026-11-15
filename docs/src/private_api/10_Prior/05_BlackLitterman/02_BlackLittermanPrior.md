```@meta
Description = "Black-Litterman Prior, private API of PortfolioOptimisers.jl: BlackLittermanViewSystem, calc_omega, bl_preroll, announce_bl_departures, bl_posteriors, …"
```

# Black-Litterman Prior: private API

## Types

```@docs
BlackLittermanViewSystem
```

## Functions

```@docs
calc_omega
bl_preroll
announce_bl_departures
bl_posteriors
bl_view_block
assert_bl_precomputed_universe
vanilla_posteriors
apply_rf
remove_excl_views
PortfolioOptimisers.assert_bl_views_axis
PortfolioOptimisers.bl_posterior
PortfolioOptimisers.fold_prior_step(pe::BlackLittermanPrior, rd::ReturnsResult, gaps::PortfolioOptimisers.Option{<:AbstractVector{Bool}})
```
