```@meta
Description = "Greedy allocation, public API of PortfolioOptimisers.jl: GreedyAllocationResult, GreedyAllocation, factory, optimise."
```

# Greedy allocation

```@docs
GreedyAllocationResult
GreedyAllocation
factory(res::GreedyAllocationResult, fb::Option{<:FOptE_FOpt_FbChain})
optimise(::GreedyAllocation{<:Any, <:Any, <:Any, Nothing}, ::FiniteAllocationInput)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
