# Bootstrap Uncertainty Sets

```@docs
StationaryBootstrap
CircularBootstrap
MovingBootstrap
ARCHUncertaintySet
ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::NumMat,
             F::Union{Nothing, <:NumMat} = nothing; dims::Int = 1, kwargs...)
ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any, <:Any,
                                    <:Any, <:Any, <:Any}, X::NumMat,
             F::Union{Nothing, <:NumMat} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, X::NumMat,
                F::Union{Nothing, <:NumMat} = nothing; dims::Int = 1, kwargs...)
mu_ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any,
                                       <:Any, <:Any, <:Any, <:Any}, X::NumMat,
                F::Union{Nothing, <:NumMat} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any,
                                       <:Any, <:Any, <:Any}, X::NumMat,
                F::Union{Nothing, <:NumMat} = nothing; dims::Int = 1, kwargs...)
sigma_ucs(ue::ARCHUncertaintySet{<:Any, <:EllipseUncertaintySetAlgorithm, <:Any,
                                          <:Any, <:Any, <:Any, <:Any}, X::NumMat,
                   F::Union{Nothing, <:NumMat} = nothing; dims::Int = 1, kwargs...)
PortfolioOptimisers.BootstrapUncertaintySetEstimator
PortfolioOptimisers.ARCHBootstrapSet
PortfolioOptimisers.bootstrap_func
PortfolioOptimisers.bootstrap_generator
PortfolioOptimisers.mu_bootstrap_generator
PortfolioOptimisers.sigma_bootstrap_generator
```
