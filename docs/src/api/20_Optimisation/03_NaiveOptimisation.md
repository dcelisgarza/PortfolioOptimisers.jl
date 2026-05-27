# Naive optimisation

```@docs
NaiveOptimisationEstimator
needs_previous_weights(opt::NaiveOptimisationEstimator)
assert_internal_optimiser(::NaiveOptimisationEstimator)
assert_external_optimiser(::NaiveOptimisationEstimator)
NaiveOptimisationResult
factory(res::NaiveOptimisationResult, fb::Option{<:OptE_Opt})
InverseVolatility
factory(opt::InverseVolatility, w::AbstractVector)
opt_view(opt::InverseVolatility, i, args...)
assert_external_optimiser(opt::InverseVolatility)
_optimise(iv::InverseVolatility, rd::ReturnsResult)
optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
EqualWeighted
factory(opt::EqualWeighted, w::AbstractVector)
opt_view(opt::EqualWeighted, i, args...)
_optimise(ew::EqualWeighted, rd::ReturnsResult)
optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)
RandomWeighted
factory(opt::RandomWeighted, w::AbstractVector)
opt_view(opt::RandomWeighted, i, args...)
_optimise(rw::RandomWeighted, rd::ReturnsResult)
optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)
```
