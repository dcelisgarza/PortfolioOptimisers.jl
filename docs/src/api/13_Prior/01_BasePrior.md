# Base Prior

```@docs
LowOrderPrior
HighOrderPrior
prior(pr::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
prior(pr::AbstractPriorResult, args...; kwargs...)
clusterise(cle::AbstractClustersEstimator, pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing, cle_pr::Bool = true, kwargs...)
Base.getproperty(obj::HighOrderPrior, sym::Symbol)
AbstractPriorEstimator
AbstractLowOrderPriorEstimator
AbstractLowOrderPriorEstimator_A
AbstractLowOrderPriorEstimator_F
AbstractLowOrderPriorEstimator_AF
AbstractLowOrderPriorEstimator_A_AF
AbstractLowOrderPriorEstimator_F_AF
AbstractLowOrderPriorEstimator_A_F_AF
AbstractHighOrderPriorEstimator
AbstractPriorResult
Pr_RR
PrE_Pr
prior_view(pr::Union{Nothing, AbstractPriorEstimator}, args...)
prior_view(pr::LowOrderPrior, rd)
prior_view(pr::HighOrderPrior, rd)
```
