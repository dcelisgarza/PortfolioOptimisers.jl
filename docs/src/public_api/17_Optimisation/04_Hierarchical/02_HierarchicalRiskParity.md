```@meta
Description = "Hierarchical Risk Parity, public API of PortfolioOptimisers.jl: HierarchicalRiskParity, port_opt_view, _optimise, needs_previous_weights."
```

# Hierarchical Risk Parity

```@docs
HierarchicalRiskParity
port_opt_view(hrp::HierarchicalRiskParity, i, X::MatNum, args...)
_optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:OptimisationRiskMeasure}, rd::ReturnsResult = ReturnsResult())
_optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:VecOptRM}, rd::ReturnsResult = ReturnsResult())
needs_previous_weights(opt::HierarchicalRiskParity)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
