```@meta
Description = "Rank-one covariance, public API of PortfolioOptimisers.jl: RankOneCovariance, cov, cor."
```

# Rank-one covariance

The principal spectral component of a short window of price relatives, scaled so its energy trades off against the centred window's: the covariance estimate of the short-term loss-control portfolio, singular by construction.

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
