```@meta
Description = "The optimiser on the partial-fit seam, public API of PortfolioOptimisers.jl: partial_fit!, optimise."
```

# The optimiser on the partial-fit seam

An optimiser takes the online step in two verbs and one forward. `partial_fit!(opt, rd)` folds the observations of a carrier into the **prior alone** and records the rest of the carrier in a [`PortfolioOptimisers.ReturnsBufferState`](@ref); `optimise(opt)` with no returns rebuilds the carrier from the state, swaps the folded prior for its read-out and runs the ordinary batch path, so everything above the prior — the clustering estimator, the constraint estimators, every uncertainty set, a meta-optimiser's inner optimisers — is fitted exactly as batch fits it. The read-out is pure, so the fallback chain walks unchanged.

```@docs
PortfolioOptimisers.partial_fit!(opt::PortfolioOptimisers.JuMPOptimisationEstimator, rd::ReturnsResult)
optimise(opt::PortfolioOptimisers.OptimisationEstimator; kwargs...)
```
