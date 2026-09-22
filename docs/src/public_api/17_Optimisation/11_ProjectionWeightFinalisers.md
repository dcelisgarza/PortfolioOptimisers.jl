```@meta
Description = "The exact projection weight finalisers, public API of PortfolioOptimisers.jl: EuclideanWeightFinaliser, EntropicWeightFinaliser."
```

# The exact projection weight finalisers

Two weight finalisers that move weights into their bounds by an exact projection, at the budget the weights carry. A bisection on one scalar finds each projection, so no solver is necessary. The Euclidean projection moves every free weight by the same amount and holds any budget and long-short bounds. The entropic projection scales every free weight by the same factor, so it keeps their ratios, and it is defined for long-only weights. The default [`IterativeWeightFinaliser`](@ref) falls back to the Euclidean projection when its loop cannot reach the bounds.

```@docs
EuclideanWeightFinaliser
EntropicWeightFinaliser
```
