# Delta Uncertainty Sets

```@docs
DeltaUncertaintySet
ucs(ue::DeltaUncertaintySet, X::NumMat,
             F::Option{<:NumMat} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::DeltaUncertaintySet, X::NumMat,
                F::Option{<:NumMat} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::DeltaUncertaintySet, X::NumMat,
                   F::Option{<:NumMat} = nothing; dims::Int = 1, kwargs...)
```
