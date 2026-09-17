```@meta
Description = "Implied Volatility, public API of PortfolioOptimisers.jl: ImpliedVolatilityRegression, ImpliedVolatilityPremium, ImpliedVolatility, cov, cor."
```

# Implied Volatility

```@docs
ImpliedVolatilityRegression
ImpliedVolatilityPremium
ImpliedVolatility
cov(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing, iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
cor(ce::ImpliedVolatility, X::MatNum; dims::Int = 1, mean = nothing, iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
cov(ce::ImpliedVolatility, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, mean = nothing, iv::MatNum, ivpa::Option{<:Num_VecNum} = nothing, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
