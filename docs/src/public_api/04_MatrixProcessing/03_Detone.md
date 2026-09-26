```@meta
Description = "Detone, public API of PortfolioOptimisers.jl: AbstractDetoneEstimator, Detone, detone, detone!."
```

# [Detone](@id api-detone)

Most assets move with the market, and that common movement dominates their correlations. Detoning removes the `n` largest eigenvalues, which carry the market movement, so the correlations that remain show how the assets relate to each other apart from the market [mlp1](@cite).

A detoned matrix can have zero or negative eigenvalues, so it does not suit an optimisation that needs a positive definite covariance. Use it where the matrix only groups the assets, as in a clustering optimisation.

```@docs
AbstractDetoneEstimator
Detone
detone
detone!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
