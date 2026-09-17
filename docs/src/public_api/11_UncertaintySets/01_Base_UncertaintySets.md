```@meta
Description = "Base Uncertainty Sets, public API of PortfolioOptimisers.jl: AbstractUncertaintySetEstimator, AbstractPriorUncertaintySetEstimator, …"
```

# Base Uncertainty Sets

```@docs
AbstractUncertaintySetEstimator
AbstractPriorUncertaintySetEstimator
AbstractUncertaintySetAlgorithm
AbstractUncertaintySetResult
AbstractUncertaintyKAlgorithm
AbstractCompactRadiusAlgorithm
AbstractUncertaintyEpsAlgorithm
AbstractUncertaintySetClass
BoxUncertaintySet
BoxUncertaintySetAlgorithm
MuUncertaintySetClass
SigmaUncertaintySetClass
NormalKUncertaintyAlgorithm
GeneralKUncertaintyAlgorithm
ChiSqKUncertaintyAlgorithm
EllipsoidalUncertaintySet
EllipsoidalUncertaintySetAlgorithm
NormBallUncertaintySetAlgorithm
ucs(uc::Option{<:Tuple{<:Option{<:AbstractUncertaintySetResult}, <:Option{<:AbstractUncertaintySetResult}}}, args...; kwargs...)
ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
ucs(ue::AbstractPriorUncertaintySetEstimator, ::AbstractPriorResult; kwargs...)
ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult, pr::AbstractPriorResult; kwargs...)
mu_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...; kwargs...)
mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult, pr::AbstractPriorResult; kwargs...)
sigma_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...; kwargs...)
sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult, pr::AbstractPriorResult; kwargs...)
reads_prior_result
k_ucs
port_opt_view(risk_ucs::Option{<:AbstractUncertaintySetEstimator}, ::Any, args...)
port_opt_view(risk_ucs::BoxUncertaintySet{<:VecNum, <:VecNum}, i, args...)
port_opt_view(risk_ucs::BoxUncertaintySet{<:MatNum, <:MatNum}, i, args...)
port_opt_view(risk_ucs::EllipsoidalUncertaintySet{<:MatNum, <:Any, <:SigmaUncertaintySetClass}, i, args...)
port_opt_view(risk_ucs::EllipsoidalUncertaintySet{<:MatNum, <:Any, <:MuUncertaintySetClass}, i, args...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
