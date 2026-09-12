# Bayesian Black-Litterman Prior

```@docs
BayesianBlackLittermanPrior
prior(pe::BayesianBlackLittermanPrior, X::MatNum, F::MatNum,
               pnl::Option{<:AssetPanel} = nothing; dims::Int = 1, strict::Bool = false,
               kwargs...)
PortfolioOptimisers.show_fields(::BayesianBlackLittermanPrior)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
