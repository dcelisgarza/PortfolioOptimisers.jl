```@meta
Description = "The algorithms of the short-term sparse portfolio, public API of PortfolioOptimisers.jl: L1Optimum, HuberOptimum, AlternatingDirectionMethod."
```

# The algorithms of the short-term sparse portfolio

The three ways a [`ShortTermSparsePortfolio`](@ref) step finds the iterate it projects: the optimum of the programme its paper states, the closed form of the fixed point of the paper's iteration, and the paper's iteration at its own stop.

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
