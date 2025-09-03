# Distance

```@docs
Distance
Distance()
distance(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:SimpleDistance}, rho::AbstractMatrix, args...; kwargs...)
cor_and_dist(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:SimpleAbsoluteDistance}, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:SimpleAbsoluteDistance}, rho::AbstractMatrix, args...; kwargs...)
cor_and_dist(::Distance{<:SimpleAbsoluteDistance}, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:LogDistance}, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:LogDistance}, ce::Union{<:LTDCovariance, <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}}, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:LogDistance}, rho::AbstractMatrix, args...; kwargs...)
cor_and_dist(::Distance{<:LogDistance}, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
cor_and_dist(::Distance{<:LogDistance}, ce::Union{<:LTDCovariance, <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}}, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(de::Distance{<:VariationInfoDistance}, ::Any, X::AbstractMatrix; dims::Int = 1, kwargs...)
cor_and_dist(de::Distance{<:VariationInfoDistance}, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:CorrelationDistance}, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:CorrelationDistance}, rho::AbstractMatrix, args...; kwargs...)
cor_and_dist(::Distance{<:CorrelationDistance}, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:CanonicalDistance}, ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:CanonicalDistance}, ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any}, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:CanonicalDistance}, ce::Union{<:LTDCovariance, <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}}, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:CanonicalDistance}, ce::Union{<:DistanceCovariance, <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}}, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:CanonicalDistance}, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
distance(::Distance{<:CanonicalDistance}, rho::AbstractMatrix, args...; kwargs...)
cor_and_dist(::Distance{<:CanonicalDistance}, ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
cor_and_dist(::Distance{<:CanonicalDistance}, ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any}, X::AbstractMatrix; dims::Int = 1, kwargs...)
cor_and_dist(::Distance{<:CanonicalDistance}, ce::Union{<:LTDCovariance, <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}}, X::AbstractMatrix; dims::Int = 1, kwargs...)
cor_and_dist(::Distance{<:CanonicalDistance}, ce::Union{<:DistanceCovariance, <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}}, X::AbstractMatrix; dims::Int = 1, kwargs...)
cor_and_dist(::Distance{<:CanonicalDistance}, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
```
