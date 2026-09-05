# Phylogeny Clustering

```@docs
_clusterise(alg::HClustAlgorithm, onc::AbstractOptimalNumberClustersEstimator,
                    S::MatNum, D::MatNum, P::MatNum; branchorder::Symbol = :optimal)
clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractTreeType,
                                                                     <:HopCount}}, X::MatNum;
                    dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
clusterise(nte::NetworkClustersEstimator{<:NetworkEstimator{<:Any, <:Any,
                                                                     <:AbstractNonNegativeSimilarityMatrixAlgorithm,
                                                                     <:HopCount}},
                    X::MatNum; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
HClE_HCl
```
