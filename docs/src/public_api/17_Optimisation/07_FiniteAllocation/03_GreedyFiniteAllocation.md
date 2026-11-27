```@meta
Description = "Greedy finite allocation, public API of PortfolioOptimisers.jl: GreedyAllocationResult, GreedyAllocation, port_opt_view, factory, _optimise."
```

# Greedy finite allocation

```@docs
GreedyAllocationResult
GreedyAllocation
port_opt_view(ga::GreedyAllocation, i, args...)
factory(res::GreedyAllocationResult, fb::Option{<:FOptE_FOpt_FbChain})
_optimise(ga::GreedyAllocation, fai::FiniteAllocationInput)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
