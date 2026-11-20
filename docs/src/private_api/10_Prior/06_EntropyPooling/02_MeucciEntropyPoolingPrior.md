```@meta
Description = "Meucci entropy pooling Prior, private API of PortfolioOptimisers.jl: VecMeucciEP, CvarViewSearch, ep_cvar_views_setup, ep_cvar_views_solve!, ep_prior, …"
```

# Meucci entropy pooling Prior: private API

## Types

```@docs
VecMeucciEP
CvarViewSearch
```

## Functions

```@docs
ep_cvar_views_setup
ep_cvar_views_solve!
ep_prior(alg::StagedEP, pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing; strict::Bool = false, kwargs...)
ep_prior(alg::H0_EntropyPooling, pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing; strict::Bool = false, kwargs...)
PortfolioOptimisers.show_fields(::MeucciEntropyPoolingPrior)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
