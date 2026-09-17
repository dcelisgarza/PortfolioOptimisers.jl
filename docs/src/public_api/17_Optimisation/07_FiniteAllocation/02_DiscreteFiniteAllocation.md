```@meta
Description = "Discrete allocation, public API of PortfolioOptimisers.jl: DiscreteAllocationResult, DiscreteAllocation, factory, optimise."
```

# Discrete allocation

```@docs
DiscreteAllocationResult
DiscreteAllocation
factory(res::DiscreteAllocationResult, fb::Option{<:FOptE_FOpt_FbChain})
optimise(::DiscreteAllocation{<:Any, <:Any, <:Any, <:Any, Nothing}, ::FiniteAllocationInput)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
