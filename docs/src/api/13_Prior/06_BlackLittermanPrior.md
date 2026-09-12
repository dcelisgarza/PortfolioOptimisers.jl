# Black-Litterman Prior

```@docs
BlackLittermanPrior
prior(pe::BlackLittermanPrior, X::MatNum,
               F::Option{<:MatNum} = nothing, pnl::Option{<:AssetPanel} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
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
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
