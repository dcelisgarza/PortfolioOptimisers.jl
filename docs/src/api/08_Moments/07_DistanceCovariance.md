# Distance Covariance

```@docs
DistanceCovariance
cov(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)
cor(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)
calc_pairwise_dists(ce::DistanceCovariance, v1::VecNum, v2::VecNum, ::Nothing)
cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum, w::Option{<:StatsBase.AbstractWeights} = nothing)
cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum, w::Option{<:StatsBase.AbstractWeights} = nothing)
cor_distance(ce::DistanceCovariance, X::MatNum, w::Option{<:StatsBase.AbstractWeights} = nothing)
cov_distance(ce::DistanceCovariance, X::MatNum, w::Option{<:StatsBase.AbstractWeights} = nothing)
```
