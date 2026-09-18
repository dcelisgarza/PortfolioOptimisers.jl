```@meta
Description = "Naive optimisation, public API of PortfolioOptimisers.jl: NaiveOptimisationResult, InverseVolatility, EqualWeighted, RandomWeighted, PreviousWeights, …"
```

# Naive optimisation

```@docs
NaiveOptimisationResult
InverseVolatility
EqualWeighted
RandomWeighted
PreviousWeights
BestConstantRebalancedPortfolio
factory(res::NaiveOptimisationResult, fb::Option{<:OptE_Opt_FbChain})
optimise(iv::InverseVolatility{<:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, kwargs...)
optimise(ew::EqualWeighted{<:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, kwargs...)
optimise(rw::RandomWeighted{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, kwargs...)
factory(pw::PreviousWeights, w::VecNum)
optimise(pw::PreviousWeights{<:Any, Nothing}, rd::ReturnsResult = ReturnsResult(); kwargs...)
optimise(bcrp::BestConstantRebalancedPortfolio{<:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, kwargs...)
_optimise(iv::InverseVolatility, rd::ReturnsResult)
_optimise(ew::EqualWeighted, rd::ReturnsResult)
_optimise(rw::RandomWeighted, rd::ReturnsResult)
_optimise(pw::PreviousWeights, rd::ReturnsResult = ReturnsResult(); kwargs...)
_optimise(bcrp::BestConstantRebalancedPortfolio, rd::ReturnsResult)
needs_previous_weights(opt::NaiveOptimisationEstimator)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
