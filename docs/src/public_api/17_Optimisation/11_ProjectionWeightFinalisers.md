```@meta
Description = "The exact projection weight finalisers, public API of PortfolioOptimisers.jl: EuclideanWeightFinaliser, EntropicWeightFinaliser."
```

# The exact projection weight finalisers

The two weight finalisers below move weights into their bounds by an exact projection, and keep the sum of the weights. Each projection is a clip of the weights at one scalar. A binary search over the kinks of that scalar finds it exactly, so no solver is necessary. The Euclidean projection moves every free weight by the same amount, and it works with any budget and with bounds on long and short weights. The entropic projection scales every free weight by the same factor, so it keeps their ratios, and it is defined for long-only weights. The default [`IterativeWeightFinaliser`](@ref) falls back to the Euclidean projection when its loop cannot reach the bounds.

```@docs
EuclideanWeightFinaliser
EntropicWeightFinaliser
```
