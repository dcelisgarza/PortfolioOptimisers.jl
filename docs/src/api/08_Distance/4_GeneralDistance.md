# General Distance

```@docs
GeneralDistance
distance(de::GeneralDistance{<:Any, <:SimpleDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
distance(de::GeneralDistance{<:Any, <:LogDistance},
                  ce::Union{<:LTDCovariance,
                            <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(de::GeneralDistance{<:Any, <:VariationInfoDistance}, ::Any,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(de::GeneralDistance{<:Any, <:SimpleDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
cor_and_dist(de::GeneralDistance{<:Any, <:SimpleDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
```
