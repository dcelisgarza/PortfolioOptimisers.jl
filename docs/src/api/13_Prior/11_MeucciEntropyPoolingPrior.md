# Meucci Entropy Pooling

```@docs
MeucciEntropyPoolingPrior
VecMeucciEP
ep_cvar_views_solve!
prior(pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
      dims::Int = 1, strict::Bool = false, kwargs...)
ep_prior(alg::StagedEP, pe::MeucciEntropyPoolingPrior, X::MatNum, F::Option{<:MatNum};
         strict::Bool = false, kwargs...)
ep_prior(alg::H0_EntropyPooling, pe::MeucciEntropyPoolingPrior, X::MatNum,
         F::Option{<:MatNum}; strict::Bool = false, kwargs...)
```
