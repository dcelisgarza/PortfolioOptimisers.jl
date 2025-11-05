# Distance

```@docs
Distance
distance(::Distance{Nothing, <:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                  X::NumMat; dims::Int = 1, kwargs...)
distance(::Distance{Nothing, <:LogDistance},
                  ce::Union{<:LowerTailDependenceCovariance,
                            <:PortfolioOptimisersCovariance{<:LowerTailDependenceCovariance, <:Any}},
                  X::NumMat; dims::Int = 1, kwargs...)
distance(de::Distance{Nothing, <:VariationInfoDistance}, ::Any, X::NumMat;
                  dims::Int = 1, kwargs...)
distance(::Distance{Nothing, <:SimpleDistance}, rho::NumMat, args...;
                  kwargs...)
cor_and_dist(::Distance{Nothing, <:SimpleDistance},
                      ce::StatsBase.CovarianceEstimator, X::NumMat; dims::Int = 1,
                      kwargs...)
distance(de::Distance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::NumMat; dims::Int = 1, kwargs...)
```
