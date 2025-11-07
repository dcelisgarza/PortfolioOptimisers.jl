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
ucs(uc::Union{Nothing,
                       <:Tuple{<:Option{<:PortfolioOptimisers.AbstractUncertaintySetResult},
                               <:Option{<:PortfolioOptimisers.AbstractUncertaintySetResult}}}, args...;
             kwargs...)
ucs(uc::PortfolioOptimisers.AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
mu_ucs(uc::Option{<:PortfolioOptimisers.AbstractUncertaintySetResult}, args...; kwargs...)
mu_ucs(uc::PortfolioOptimisers.AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
sigma_ucs(uc::Option{<:PortfolioOptimisers.AbstractUncertaintySetResult}, args...; kwargs...)
PortfolioOptimisers.AbstractUncertaintySetEstimator
PortfolioOptimisers.AbstractUncertaintySetAlgorithm
PortfolioOptimisers.AbstractUncertaintySetResult
PortfolioOptimisers.AbstractUncertaintyKAlgorithm
PortfolioOptimisers.AbstractEllipseUncertaintySetResultClass
PortfolioOptimisers.ucs_factory
PortfolioOptimisers.k_ucs
```
