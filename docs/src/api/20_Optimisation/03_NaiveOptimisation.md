# Naive optimisation

```@docs
NaiveOptimisationEstimator
NaiveOptimisationResult
InverseVolatility
EqualWeighted
RandomWeighted
optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)
optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult; dims::Int = 1, kwargs...)
```
