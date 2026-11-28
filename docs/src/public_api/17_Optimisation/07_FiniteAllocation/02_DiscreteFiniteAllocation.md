```@meta
Description = "Discrete finite allocation, public API of PortfolioOptimisers.jl: DiscreteAllocationResult, DiscreteAllocation, port_opt_view, factory, _optimise."
```

# Discrete finite allocation

```@docs
DiscreteAllocationResult
DiscreteAllocation
port_opt_view(da::DiscreteAllocation, i, args...)
factory(res::DiscreteAllocationResult, fb::Option{<:FOptE_FOpt_FbChain})
_optimise(da::DiscreteAllocation, fai::FiniteAllocationInput)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
