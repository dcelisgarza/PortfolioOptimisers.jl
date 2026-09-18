```@meta
Description = "Hierarchical Risk Parity, public API of PortfolioOptimisers.jl: HierarchicalRiskParity, port_opt_view, optimise, needs_previous_weights."
```

# Hierarchical Risk Parity

```@docs
HierarchicalRiskParity
port_opt_view(hrp::HierarchicalRiskParity, i, X::MatNum, args...)
optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing}, rd::ReturnsResult; dims::Int = 1, kwargs...)
needs_previous_weights(opt::HierarchicalRiskParity)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
