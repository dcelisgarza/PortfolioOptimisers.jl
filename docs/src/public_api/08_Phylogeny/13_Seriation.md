```@meta
Description = "Seriation, public API of PortfolioOptimisers.jl: AbstractSeriationEstimator, AbstractSeriationResult, HierarchicalSeriation, HierarchicalSeriationResult, …"
```

# Seriation

```@docs
AbstractSeriationEstimator
AbstractSeriationResult
HierarchicalSeriation
HierarchicalSeriationResult
FixedSeriationResult
seriate(ser::HierarchicalSeriation, X::MatNum; dims::Int = 1, kwargs...)
port_opt_view(ser::FixedSeriationResult, i, args...)
```
