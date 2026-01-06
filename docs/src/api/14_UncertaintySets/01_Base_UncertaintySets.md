# Base Uncertainty Sets

```@docs
BoxUncertaintySet
BoxUncertaintySetAlgorithm
MuEllipsoidalUncertaintySet
SigmaEllipsoidalUncertaintySet
NormalKUncertaintyAlgorithm
GeneralKUncertaintyAlgorithm
ChiSqKUncertaintyAlgorithm
EllipsoidalUncertaintySet
EllipsoidalUncertaintySetAlgorithm
ucs(uc::Option{<:Tuple{<:Option{<:AbstractUncertaintySetResult},
                       <:Option{<:AbstractUncertaintySetResult}}}, args...;
             kwargs...)
ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
mu_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...; kwargs...)
mu_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
sigma_ucs(uc::Option{<:AbstractUncertaintySetResult}, args...; kwargs...)
AbstractUncertaintySetEstimator
AbstractUncertaintySetAlgorithm
AbstractUncertaintySetResult
AbstractUncertaintyKAlgorithm
AbstractEllipsoidalUncertaintySetResultClass
ucs_selector
k_ucs
```
