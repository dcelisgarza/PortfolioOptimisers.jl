# Distance Covariance

```@docs
DistanceCovariance
cov(ce::DistanceCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
cor(ce::DistanceCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
PortfolioOptimisers.cor_distance(ce::DistanceCovariance, v1::AbstractVector, v2::AbstractVector)
PortfolioOptimisers.cov_distance(ce::DistanceCovariance, v1::AbstractVector, v2::AbstractVector)
PortfolioOptimisers.cor_distance(ce::DistanceCovariance, X::AbstractMatrix)
PortfolioOptimisers.cov_distance(ce::DistanceCovariance, X::AbstractMatrix)
```
