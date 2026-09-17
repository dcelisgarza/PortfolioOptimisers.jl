```@meta
Description = "Gerber Information Quality Covariance, public API of PortfolioOptimisers.jl: AssetVolatilityGerberIQScaler, ExpGerberIQDecay, BasicGerberIQ, …"
```

# Gerber Information Quality Covariance

```@docs
AssetVolatilityGerberIQScaler
ExpGerberIQDecay
BasicGerberIQ
PartialGerberIQ
FullGerberIQ
GerberIQCovariance
cor(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)
cov(ce::GerberIQCovariance, X::MatNum; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
