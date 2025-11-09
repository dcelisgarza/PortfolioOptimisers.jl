# Distance Covariance

```@docs
DistanceCovariance
cov(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)
cor(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)
PortfolioOptimisers.cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)
PortfolioOptimisers.cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)
PortfolioOptimisers.cor_distance(ce::DistanceCovariance, X::MatNum)
PortfolioOptimisers.cov_distance(ce::DistanceCovariance, X::MatNum)
```
