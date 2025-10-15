# Delta Uncertainty Sets

```@docs
DeltaUncertaintySet
ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
             F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
                F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::DeltaUncertaintySet, X::AbstractMatrix,
                   F::Union{Nothing, <:AbstractMatrix} = nothing; dims::Int = 1, kwargs...)
```
