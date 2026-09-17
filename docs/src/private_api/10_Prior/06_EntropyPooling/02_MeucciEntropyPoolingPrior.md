```@meta
Description = "Meucci Entropy Pooling, private API of PortfolioOptimisers.jl: VecMeucciEP, ep_cvar_views_setup, ep_cvar_views_solve!, ep_prior, show_fields."
```

# Meucci Entropy Pooling: private API

```@docs
VecMeucciEP
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
