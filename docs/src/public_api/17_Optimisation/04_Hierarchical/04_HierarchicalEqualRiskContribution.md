```@meta
Description = "Hierarchical Equal Risk Contribution, public API of PortfolioOptimisers.jl: HierarchicalEqualRiskContribution, factory, port_opt_view, optimise."
```

# Hierarchical Equal Risk Contribution

```@docs
HierarchicalEqualRiskContribution
factory(hec::HierarchicalEqualRiskContribution, w::AbstractVector)
port_opt_view(hec::HierarchicalEqualRiskContribution, i, X::MatNum, args...)
optimise(hec::HierarchicalEqualRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, branchorder::Symbol = :optimal, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
