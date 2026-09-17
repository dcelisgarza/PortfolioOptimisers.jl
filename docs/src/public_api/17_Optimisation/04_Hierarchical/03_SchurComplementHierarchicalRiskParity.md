```@meta
Description = "Schur Complement Hierarchical Risk Parity, public API of PortfolioOptimisers.jl: NonMonotonicSchurComplement, MonotonicSchurComplement, …"
```

# Schur Complement Hierarchical Risk Parity

```@docs
NonMonotonicSchurComplement
MonotonicSchurComplement
SchurComplementParams
SchurComplementHierarchicalRiskParityResult
SchurComplementHierarchicalRiskParity
port_opt_view(sp::SchurComplementParams, i, X::MatNum, args...)
factory(res::SchurComplementHierarchicalRiskParityResult, fb::Option{<:OptE_Opt_FbChain})
port_opt_view(sh::SchurComplementHierarchicalRiskParity, i, X::MatNum, args...)
optimise(sh::SchurComplementHierarchicalRiskParity{<:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, kwargs...)
needs_previous_weights(opt::SchurComplementHierarchicalRiskParity)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
