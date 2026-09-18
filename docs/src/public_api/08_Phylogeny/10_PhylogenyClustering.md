```@meta
Description = "Phylogeny Clustering, public API of PortfolioOptimisers.jl: clusterise."
```

# Phylogeny Clustering

```@docs
clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any, <:AbstractTreeType, <:HopCount}}, X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any, <:AbstractNonNegativeSimilarityMatrixAlgorithm, <:HopCount}}, X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
```
