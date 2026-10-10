```@meta
Description = "Sparse portfolio algorithms, public API of PortfolioOptimisers.jl: L1Optimum, HuberOptimum, AlternatingDirectionMethod, sparse_portfolio_iterate."
```

# Sparse portfolio algorithms

A [`ShortTermSparsePortfolio`](@ref) step finds the point it projects onto the allowed weights in one of three ways, each from the short-term sparse portfolio of [lai2018sspo](@citet). `L1Optimum` takes the optimum of the problem of [lai2018sspo](@cite). `HuberOptimum` takes the fixed point of the iteration of [lai2018sspo](@cite), in closed form. `AlternatingDirectionMethod` runs that iteration until the stopping rule of [lai2018sspo](@cite). A new algorithm subtypes `AbstractSparsePortfolioAlgorithm` and adds a method of `sparse_portfolio_iterate`.

```@docs
L1Optimum
HuberOptimum
AlternatingDirectionMethod
PortfolioOptimisers.sparse_portfolio_iterate
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
