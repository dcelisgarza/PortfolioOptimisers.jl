# Distance

```@docs
Distance
distance(::Distance{Nothing, <:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{Nothing, <:LogDistance},
                  ce::Union{<:LowerTailDependenceCovariance,
                            <:PortfolioOptimisersCovariance{<:LowerTailDependenceCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(de::Distance{Nothing, <:VariationInfoDistance}, ::Any, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
distance(::Distance{Nothing, <:SimpleDistance}, rho::AbstractMatrix, args...;
                  kwargs...)
cor_and_dist(::Distance{Nothing, <:SimpleDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
distance(de::Distance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
```
