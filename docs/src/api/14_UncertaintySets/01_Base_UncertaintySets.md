# Base Uncertainty Sets

```@docs
BoxUncertaintySet
BoxUncertaintySetAlgorithm
MuEllipseUncertaintySet
SigmaEllipseUncertaintySet
NormalKUncertaintyAlgorithm
GeneralKUncertaintyAlgorithm
ChiSqKUncertaintyAlgorithm
EllipseUncertaintySet
EllipseUncertaintySetAlgorithm
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
AbstractEllipseUncertaintySetResultClass
ucs_selector
k_ucs
```
