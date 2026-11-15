```@meta
Description = "Hierarchical Equal Risk Contribution, public API of PortfolioOptimisers.jl: HierarchicalEqualRiskContribution, factory, port_opt_view, _optimise, …"
```

# Hierarchical Equal Risk Contribution

```@docs
HierarchicalEqualRiskContribution
factory(hec::HierarchicalEqualRiskContribution, w::AbstractVector)
port_opt_view(hec::HierarchicalEqualRiskContribution, i, X::MatNum, args...)
_optimise(hec::HierarchicalEqualRiskContribution, rd::ReturnsResult = ReturnsResult())
needs_previous_weights(opt::HierarchicalEqualRiskContribution)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
