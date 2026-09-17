```@meta
Description = "Near optimal centering, public API of PortfolioOptimisers.jl: ConstrainedNearOptimalCentering, UnconstrainedNearOptimalCentering, …"
```

# Near optimal centering

```@docs
ConstrainedNearOptimalCentering
UnconstrainedNearOptimalCentering
NearOptimalCenteringResult
NearOptimalCentering
factory(res::NearOptimalCenteringResult, fb::Option{<:OptE_Opt_FbChain})
Base.getproperty(r::NearOptimalCenteringResult, sym::Symbol)
factory(noc::NearOptimalCentering, w::AbstractVector)
port_opt_view(noc::NearOptimalCentering, i, X::MatNum, args...)
optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
needs_previous_weights(opt::NearOptimalCentering)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
