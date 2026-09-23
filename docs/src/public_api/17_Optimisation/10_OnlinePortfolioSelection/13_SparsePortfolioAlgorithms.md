```@meta
Description = "The algorithms of the short-term sparse portfolio, public API of PortfolioOptimisers.jl: L1Optimum, HuberOptimum, AlternatingDirectionMethod."
```

# The algorithms of the short-term sparse portfolio

A [`ShortTermSparsePortfolio`](@ref) step finds the point it projects onto the allowed weights in one of three ways, each from the paper of Lai, Yang, Fang and Wu (2018). `L1Optimum` takes the optimum of the problem that the paper states. `HuberOptimum` takes the fixed point of the paper's iteration, in closed form. `AlternatingDirectionMethod` runs that iteration until the stopping rule of the paper.

```@docs
L1Optimum
HuberOptimum
AlternatingDirectionMethod
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
