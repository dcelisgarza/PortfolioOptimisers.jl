```@meta
Description = "Naive optimisation, private API of PortfolioOptimisers.jl: NaiveOptimisationEstimator, needs_previous_weights, is_time_dependent, …"
```

# Naive optimisation: private API

```@docs
NaiveOptimisationEstimator
needs_previous_weights(opt::NaiveOptimisationEstimator)
is_time_dependent(opt::NaiveOptimisationEstimator)
reset_time_dependent_estimator(opt::NaiveOptimisationEstimator)
assert_internal_optimiser(::NaiveOptimisationEstimator)
assert_external_optimiser(::NaiveOptimisationEstimator)
naive_optimiser_td_defaults
assert_external_optimiser(opt::InverseVolatility)
_optimise(iv::InverseVolatility, rd::ReturnsResult)
_optimise(ew::EqualWeighted, rd::ReturnsResult)
_optimise(rw::RandomWeighted, rd::ReturnsResult)
_optimise(pw::PreviousWeights, rd::ReturnsResult = ReturnsResult(); kwargs...)
failed_hold_weights
```
