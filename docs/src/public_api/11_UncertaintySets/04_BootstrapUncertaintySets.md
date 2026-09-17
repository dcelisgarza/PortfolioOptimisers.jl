```@meta
Description = "Bootstrap Uncertainty Sets, public API of PortfolioOptimisers.jl: StationaryBootstrap, CircularBootstrap, MovingBootstrap, ARCHUncertaintySet, ucs, mu_ucs, …"
```

# Bootstrap Uncertainty Sets

```@docs
StationaryBootstrap
CircularBootstrap
MovingBootstrap
ARCHUncertaintySet
ucs(ue::ARCHUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:EllipsoidalUncertaintySetAlgorithm, <:Any, <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:NormBallUncertaintySetAlgorithm, <:Any, <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:NormBallUncertaintySetAlgorithm, <:Any, <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::ARCHUncertaintySet{Nothing, <:Any, <:Any, <:NormBallUncertaintySetAlgorithm, <:Any, <:Any, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
