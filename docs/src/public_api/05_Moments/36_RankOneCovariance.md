```@meta
Description = "Rank-one covariance, public API of PortfolioOptimisers.jl: RankOneCovariance, cov, cor."
```

# Rank-one covariance

`RankOneCovariance` is the covariance estimate of the short-term loss-control portfolio. It keeps only the first principal component of a short window of price relatives, and scales it against the total variance of the centred window. The matrix has rank one, so it is positive semi-definite and always singular.

```@docs
RankOneCovariance
cov(::RankOneCovariance, X::MatNum; dims::Int = 1, kwargs...)
cor(::RankOneCovariance, X::MatNum; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
