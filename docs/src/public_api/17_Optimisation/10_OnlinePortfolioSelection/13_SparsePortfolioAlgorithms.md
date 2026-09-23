```@meta
Description = "The algorithms of the short-term sparse portfolio, public API of PortfolioOptimisers.jl: L1Optimum, HuberOptimum, AlternatingDirectionMethod, …"
```

# The algorithms of the short-term sparse portfolio

A [`ShortTermSparsePortfolio`](@ref) step finds the point it projects onto the allowed weights in one of three ways, each from the paper of Lai, Yang, Fang and Wu (2018). `L1Optimum` takes the optimum of the problem that the paper states. `HuberOptimum` takes the fixed point of the paper's iteration, in closed form. `AlternatingDirectionMethod` runs that iteration until the stopping rule of the paper. A new algorithm subtypes `AbstractSparsePortfolioAlgorithm` and adds a method of `sparse_portfolio_iterate`.

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
