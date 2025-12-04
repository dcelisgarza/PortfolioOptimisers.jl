# Distance Covariance

```@docs
DistanceCovariance
cov(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)
cor(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)
cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)
cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)
cor_distance(ce::DistanceCovariance, X::MatNum)
cov_distance(ce::DistanceCovariance, X::MatNum)
```
