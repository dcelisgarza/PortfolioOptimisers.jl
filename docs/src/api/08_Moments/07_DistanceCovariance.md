# Distance Covariance

```@docs
DistanceCovariance
cov(ce::DistanceCovariance, X::NumMat; dims::Int = 1, kwargs...)
cor(ce::DistanceCovariance, X::NumMat; dims::Int = 1, kwargs...)
PortfolioOptimisers.cor_distance(ce::DistanceCovariance, v1::NumVec, v2::NumVec)
PortfolioOptimisers.cov_distance(ce::DistanceCovariance, v1::NumVec, v2::NumVec)
PortfolioOptimisers.cor_distance(ce::DistanceCovariance, X::NumMat)
PortfolioOptimisers.cov_distance(ce::DistanceCovariance, X::NumMat)
```
