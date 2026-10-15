```@meta
Description = "Prior Returns Source, public API of PortfolioOptimisers.jl: clusterise, returns_source."
```

# [Prior Returns Source](@id api-prior-returns-source)

A clustering, phylogeny or centrality estimator reads the asset returns of a prior result or of the returns data, in the source that it states in its `x_src` field. The [private page](@ref private-api-prior-returns-source) documents the picker that gives it the matrix. A prior of the library states its own source, and [`returns_source`](@ref) gives it to an optimiser.

```@docs
clusterise(cle::AbstractClustersEstimator, pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing, iv::Option{<:MatNum} = nothing, kwargs...)
PortfolioOptimisers.returns_source(::Union{<:EmpiricalPrior, <:FactorPrior, <:HighOrderPriorEstimator, <:HighOrderFactorPriorEstimator, <:BlackLittermanPrior, <:BayesianBlackLittermanPrior, <:FactorBlackLittermanPrior, <:AugmentedBlackLittermanPrior, <:EntropyPoolingPrior, <:MeucciEntropyPoolingPrior, <:OpinionPoolingPrior, <:CrossSectionalFactorPrior, <:LowDimensionEnsemblePrior})
```
