```@meta
Description = "Entropy pooling Prior (c), private API of PortfolioOptimisers.jl: VecEP, show_fields, ep_prior, embedded_pin_weights."
```

# Entropy pooling Prior (c): private API

```@docs
VecEP
PortfolioOptimisers.show_fields(::EntropyPoolingPrior)
ep_prior(alg::StagedEP, pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing; strict::Bool = false, kwargs...)
ep_prior(alg::H0_EntropyPooling, pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum}, pnl::Option{<:AssetPanel} = nothing; strict::Bool = false, kwargs...)
embedded_pin_weights
```
