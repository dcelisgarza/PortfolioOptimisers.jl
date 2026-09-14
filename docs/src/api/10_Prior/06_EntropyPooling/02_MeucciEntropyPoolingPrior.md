# Meucci Entropy Pooling

```@docs
MeucciEntropyPoolingPrior
VecMeucciEP
ep_cvar_views_setup
ep_cvar_views_solve!
prior(pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing,
      pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false, kwargs...)
ep_prior(alg::StagedEP, pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum},
         pnl::Option{<:AssetPanel} = nothing; strict::Bool = false, kwargs...)
ep_prior(alg::H0_EntropyPooling, pe::MeucciEntropyPoolingPrior, X::MatNum,
         F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing; strict::Bool = false,
         kwargs...)
PortfolioOptimisers.show_fields(::MeucciEntropyPoolingPrior)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
