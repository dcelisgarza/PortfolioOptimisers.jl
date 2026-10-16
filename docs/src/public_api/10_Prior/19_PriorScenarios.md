```@meta
Description = "Prior Scenarios, public API of PortfolioOptimisers.jl: keeps_observations."
```

# Prior Scenarios

A prior result carries its scenarios in `X`. A prior that reweights observations, such as an entropy pooling prior, states its prior probabilities on the scenarios of the prior that it wraps. `keeps_observations` tells it when the wrapped prior answers one scenario for each observation, so that it reads the count of the scenarios with no fit. A new prior implements a method of it when no configuration and no data makes its result drop a row. The other names of this topic are private, and the [private page](@ref private-api-prior-scenarios) documents them.

```@docs
PortfolioOptimisers.keeps_observations(::PortfolioOptimisers.AbstractPriorEstimator)
```
