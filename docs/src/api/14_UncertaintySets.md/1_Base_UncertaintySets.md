# Base Uncertainty Sets

```@docs
BoxUncertaintySetAlgorithm
BoxUncertaintySet
EllipseUncertaintySetAlgorithm
EllipseUncertaintySet
NormalKUncertaintyAlgorithm
GeneralKUncertaintyAlgorithm
ChiSqKUncertaintyAlgorithm
ucs(uc::Union{Nothing,
                       <:Tuple{<:Union{Nothing, <:PortfolioOptimisers.AbstractUncertaintySetResult},
                               <:Union{Nothing, <:PortfolioOptimisers.AbstractUncertaintySetResult}}}, args...;
             kwargs...)
mu_ucs(uc::Union{Nothing, <:PortfolioOptimisers.AbstractUncertaintySetResult}, args...; kwargs...)
sigma_ucs(uc::Union{Nothing, <:PortfolioOptimisers.AbstractUncertaintySetResult}, args...; kwargs...)
ucs(uc::PortfolioOptimisers.AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
mu_ucs(uc::PortfolioOptimisers.AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
sigma_ucs(uc::PortfolioOptimisers.AbstractUncertaintySetEstimator, rd::ReturnsResult; kwargs...)
PortfolioOptimisers.AbstractUncertaintySetEstimator
PortfolioOptimisers.AbstractUncertaintySetAlgorithm
PortfolioOptimisers.AbstractUncertaintySetResult
PortfolioOptimisers.AbstractUncertaintyKAlgorithm
PortfolioOptimisers.k_ucs
```
