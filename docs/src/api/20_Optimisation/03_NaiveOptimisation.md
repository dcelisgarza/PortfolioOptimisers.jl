# Naive optimisation

```@docs
NaiveOptimisationEstimator
needs_previous_weights(opt::NaiveOptimisationEstimator)
is_time_dependent(opt::NaiveOptimisationEstimator)
update_time_dependent_estimator(opt::NaiveOptimisationEstimator, ctx::TimeDependentContext, all_binds::Bool = true)
reset_time_dependent_estimator(opt::NaiveOptimisationEstimator)
assert_internal_optimiser(::NaiveOptimisationEstimator)
assert_external_optimiser(::NaiveOptimisationEstimator)
NaiveOptimisationResult
factory(res::NaiveOptimisationResult, fb::Option{<:OptE_Opt})
InverseVolatility
assert_external_optimiser(opt::InverseVolatility)
_optimise(iv::InverseVolatility, rd::ReturnsResult)
optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
EqualWeighted
_optimise(ew::EqualWeighted, rd::ReturnsResult)
optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)
RandomWeighted
_optimise(rw::RandomWeighted, rd::ReturnsResult)
optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)
```
