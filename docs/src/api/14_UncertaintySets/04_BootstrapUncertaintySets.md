# Bootstrap Uncertainty Sets

```@docs
StationaryBootstrap
CircularBootstrap
MovingBootstrap
ARCHUncertaintySet
ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any,
                                       <:Any, <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any,
                                          <:Any, <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
BootstrapUncertaintySetEstimator
ARCHBootstrapSet
bootstrap_indices
bootstrap_generator
mu_bootstrap_generator
sigma_bootstrap_generator
```
