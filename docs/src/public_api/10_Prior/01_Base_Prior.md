```@meta
Description = "Base Prior, public API of PortfolioOptimisers.jl: AbstractPriorEstimator, AbstractPriorResult, LowOrderPrior, HighOrderPrior, prior, forward_prior, …"
```

# Base Prior

```@docs
AbstractPriorEstimator
AbstractPriorResult
LowOrderPrior
HighOrderPrior
prior(pe::AbstractPriorEstimator, rd::ReturnsResult; kwargs...)
prior(pr::AbstractPriorResult, args...; kwargs...)
forward_prior
reconstruct_prior
clusterise(cle::AbstractClustersEstimator, pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing, x_src::Symbol = :prior, kwargs...)
port_opt_view(pr::Union{Nothing, AbstractPriorEstimator}, ::Any, args...)
port_opt_view(pr::LowOrderPrior, rd, args...)
port_opt_view(pr::HighOrderPrior, rd, args...)
```
