```@meta
Description = "Distance Covariance, private API of PortfolioOptimisers.jl: calc_pairwise_dists, calc_centred_dists, calc_dcov2, cor_distance, cov_distance."
```

# Distance Covariance: private API

```@docs
calc_pairwise_dists(ce::DistanceCovariance, v1::VecNum, v2::VecNum)
calc_centred_dists(a::MatNum, ::Nothing)
calc_dcov2(A::MatNum, B::MatNum, ::Nothing)
cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum, w::Option{<:StatsBase.AbstractWeights} = nothing)
cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum, w::Option{<:StatsBase.AbstractWeights} = nothing)
cor_distance(ce::DistanceCovariance, X::MatNum, w::Option{<:StatsBase.AbstractWeights} = nothing)
cov_distance(ce::DistanceCovariance, X::MatNum, w::Option{<:StatsBase.AbstractWeights} = nothing)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
