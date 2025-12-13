# Bootstrap Uncertainty Sets

```@docs
StationaryBootstrap
CircularBootstrap
MovingBootstrap
ARCHUncertaintySet
ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::MatNum,
             F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any,
                                       <:Any, <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, X::MatNum,
                F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any,
                                          <:Any, <:Any, <:Any, <:Any}, X::MatNum,
                   F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
BootstrapUncertaintySetEstimator
ARCHBootstrapSet
bootstrap_func
bootstrap_generator
mu_bootstrap_generator
sigma_bootstrap_generator
```
