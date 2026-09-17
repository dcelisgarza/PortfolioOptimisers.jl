```@meta
Description = "Hierarchical Risk Parity, public API of PortfolioOptimisers.jl: HierarchicalRiskParity, port_opt_view, optimise."
```

# Hierarchical Risk Parity

```@docs
HierarchicalRiskParity
port_opt_view(hrp::HierarchicalRiskParity, i, X::MatNum, args...)
optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing}, rd::ReturnsResult; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
