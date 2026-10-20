```@meta
Description = "Rank covariance, public API of PortfolioOptimisers.jl: KendallCovariance, SpearmanCovariance, cor."
```

# Rank covariance

```@docs
KendallCovariance
SpearmanCovariance
cor(::KendallCovariance, X::MatNum; dims::Int = 1, kwargs...)
cor(::SpearmanCovariance, X::MatNum; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
