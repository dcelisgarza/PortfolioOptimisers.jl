```@meta
Description = "The optimiser on the partial-fit seam, public API of PortfolioOptimisers.jl: partial_fit!, optimise."
```

# The optimiser on the partial-fit seam

An optimiser updates with two calls. `partial_fit!(opt, rd)` adds the observations of a `ReturnsResult` to the prior only, and stores the rest of the `ReturnsResult` in a [`PortfolioOptimisers.ReturnsBufferState`](@ref). `optimise(opt)` with no returns rebuilds the `ReturnsResult` from the state, replaces the prior with its current result, and runs the ordinary batch optimisation. Every part that uses the prior is then fitted as in a batch run, such as the clustering estimator, the constraint estimators, each uncertainty set and the inner optimisers of a meta-optimiser. Reading the prior does not change its state, so a fallback optimiser, which runs when the first one fails, reads the same prior.

```@docs
PortfolioOptimisers.partial_fit!(opt::PortfolioOptimisers.JuMPOptimisationEstimator, rd::ReturnsResult)
optimise(opt::PortfolioOptimisers.OptimisationEstimator; kwargs...)
```
