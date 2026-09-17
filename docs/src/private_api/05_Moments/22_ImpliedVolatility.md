```@meta
Description = "Implied Volatility, private API of PortfolioOptimisers.jl: ImpliedVolatilityAlgorithm, realised_vol, implied_vol, predict_realised_vols, …"
```

# Implied Volatility: private API

```@docs
ImpliedVolatilityAlgorithm
realised_vol
implied_vol
predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any, ivpa::Nothing)
predict_realised_vols(::ImpliedVolatilityPremium, iv::MatNum, ::Any, ivpa::Num_VecNum)
predict_realised_vols(alg::ImpliedVolatilityRegression, iv::MatNum, X::MatNum, ::Any)
PortfolioOptimisers.coverage_reduced_ivpa
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
