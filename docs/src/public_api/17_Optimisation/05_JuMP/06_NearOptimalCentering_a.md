```@meta
Description = "Near optimal centering (a), public API of PortfolioOptimisers.jl: ConstrainedNearOptimalCentering, UnconstrainedNearOptimalCentering, …"
```

# Near optimal centering (a)

```@docs
ConstrainedNearOptimalCentering
UnconstrainedNearOptimalCentering
NearOptimalCenteringResult
NearOptimalCentering
factory(res::NearOptimalCenteringResult, fb::Option{<:OptE_Opt_FbChain})
Base.getproperty(r::NearOptimalCenteringResult, sym::Symbol)
factory(noc::NearOptimalCentering, w::AbstractVector)
port_opt_view(noc::NearOptimalCentering, i, X::MatNum, args...)
needs_previous_weights(opt::NearOptimalCentering)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
