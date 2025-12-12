# Delta Uncertainty Sets

```@docs
DeltaUncertaintySet
ucs(ue::DeltaUncertaintySet, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::DeltaUncertaintySet, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::DeltaUncertaintySet, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
```
