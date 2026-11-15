```@meta
Description = "Stacking, public API of PortfolioOptimisers.jl: StackingResult, Stacking, factory, port_opt_view, _optimise, needs_previous_weights."
```

# Stacking

```@docs
StackingResult
Stacking
factory(res::StackingResult, fb::Option{<:OptE_Opt_FbChain})
factory(st::Stacking, w::AbstractVector)
port_opt_view(st::Stacking, i, X::MatNum, args...)
_optimise(st::Stacking, rd::ReturnsResult)
needs_previous_weights(opt::Stacking)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
