# Distance

```@docs
Distance
distance(::Distance{Nothing, <:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                  X::MatNum; dims::Int = 1, kwargs...)
distance(::Distance{Nothing, <:LogDistance},
                  ce::LTDCov_PLTDCov,
                  X::MatNum; dims::Int = 1, kwargs...)
distance(de::Distance{Nothing, <:VariationInfoDistance}, ::Any, X::MatNum;
                  dims::Int = 1, kwargs...)
distance(::Distance{Nothing, <:SimpleDistance}, rho::MatNum, args...;
                  kwargs...)
cor_and_dist(::Distance{Nothing, <:SimpleDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
distance(de::Distance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::MatNum; dims::Int = 1, kwargs...)
```
