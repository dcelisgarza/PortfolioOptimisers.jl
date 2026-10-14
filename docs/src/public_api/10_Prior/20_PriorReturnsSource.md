```@meta
Description = "Prior Returns Source, public API of PortfolioOptimisers.jl: clusterise."
```

# Prior Returns Source

A clustering, phylogeny or centrality estimator reads the asset returns of a prior result or of the returns data, in the source that it states in its `x_src` field. The [private page](@ref private-api-prior-returns-source) documents the picker that gives it the matrix.

```@docs
clusterise(cle::AbstractClustersEstimator, pr::Pr_RR; rd::Option{<:ReturnsResult} = nothing, iv::Option{<:MatNum} = nothing, kwargs...)
```
