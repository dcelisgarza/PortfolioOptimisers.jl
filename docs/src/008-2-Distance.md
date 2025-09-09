# Distance

```@docs
Distance
distance(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:LogDistance},
                  ce::Union{<:LTDCovariance,
                            <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(de::Distance{<:VariationInfoDistance}, ::Any, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
distance(::Distance{<:SimpleDistance}, rho::AbstractMatrix, args...; kwargs...)
cor_and_dist(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
```
