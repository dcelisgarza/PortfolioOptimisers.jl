```@meta
Description = "Schur Complement Hierarchical Risk Parity, public API of PortfolioOptimisers.jl: SchurComplementAlgorithm, NonMonotonicSchurComplement, …"
```

# Schur Complement Hierarchical Risk Parity

A new Schur complement algorithm subtypes `SchurComplementAlgorithm` and adds a method of `schur_complement_weights`.

```@docs
PortfolioOptimisers.SchurComplementAlgorithm
NonMonotonicSchurComplement
MonotonicSchurComplement
schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt, wb::WeightBounds, params::SchurComplementParams{<:Any, <:Any, <:Any, <:NonMonotonicSchurComplement, <:Any}, gamma::Option{<:Number} = nothing)
schur_complement_weights(pr::AbstractPriorResult, items::VecVecInt, wb::WeightBounds, params::SchurComplementParams{<:Any, <:Any, <:Any, <:MonotonicSchurComplement, <:Any})
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
