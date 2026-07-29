# Feature Prior

A feature prior attaches an `assets × features` matrix to the prior it wraps, without touching a single moment, so any prior becomes a source for [`FeatureDistance`](@ref).

```@docs
AbstractFeatureMatrixEstimator
feature_matrix
RegressionFeatures
AbstractPhylogenyFeatureAlgorithm
BinaryNeighbourhood
GradedNeighbourhood
phylogeny_features
PhylogenyFeatures
AssetSetsFeatures
FeaturePrior
PortfolioOptimisers.feature_estimator_view
port_opt_view(pe::FeaturePrior, i, args...)
prior(pe::FeaturePrior, X::MatNum, F::Option{<:MatNum} = nothing; dims::Int = 1,
               kwargs...)
```
