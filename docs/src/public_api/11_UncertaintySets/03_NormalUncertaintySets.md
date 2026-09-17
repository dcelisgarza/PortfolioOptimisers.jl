```@meta
Description = "Normal Uncertainty Sets, public API of PortfolioOptimisers.jl: NormalUncertaintySet, ucs, mu_ucs, sigma_ucs."
```

# Normal Uncertainty Sets

```@docs
NormalUncertaintySet
ucs(ue::NormalUncertaintySet, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:BoxUncertaintySetAlgorithm, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:ChiSqKUncertaintyAlgorithm, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:EllipsoidalUncertaintySetAlgorithm{<:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
mu_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:NormalKUncertaintyAlgorithm, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
sigma_ucs(ue::NormalUncertaintySet{Nothing, <:NormBallUncertaintySetAlgorithm{<:Any, <:Any, <:Any}, <:Any, <:Any, <:Any}, pr::AbstractPriorResult; rd = nothing, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
