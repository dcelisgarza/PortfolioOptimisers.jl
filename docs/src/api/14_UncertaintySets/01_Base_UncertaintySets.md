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
sigma_ucs(uc::AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
AbstractUncertaintySetEstimator
AbstractUncertaintySetAlgorithm
AbstractUncertaintySetResult
AbstractUncertaintyKAlgorithm
AbstractUncertaintyEpsAlgorithm
UcSE_UcS
Num_UcSK
Num_UcSEps
AbstractEllipsoidalUncertaintySetResultClass
ucs_selector
k_ucs
port_opt_view(risk_ucs::Option{<:AbstractUncertaintySetEstimator}, ::Any, args...)
port_opt_view(risk_ucs::BoxUncertaintySet{<:VecNum, <:VecNum}, i, args...)
port_opt_view(risk_ucs::BoxUncertaintySet{<:MatNum, <:MatNum}, i, args...)
port_opt_view(risk_ucs::EllipsoidalUncertaintySet{<:MatNum, <:Any, <:SigmaEllipsoidalUncertaintySet}, i, args...)
port_opt_view(risk_ucs::EllipsoidalUncertaintySet{<:MatNum, <:Any, <:MuEllipsoidalUncertaintySet}, i, args...)
vec_quantile_bounds
ellipsoidal_set
box_quantile_bounds
```
