```@meta
Description = "Entropy pooling tail view solve, private API of PortfolioOptimisers.jl: AbstractEntropyPoolingTailView, VecEPTV, EvarDualMinimum, RlvarShiftMinimum, …"
```

# [Entropy pooling tail view solve: private API](@id private-api-entropy-pooling-tail-view-solve)

## Types

```@docs
AbstractEntropyPoolingTailView
VecEPTV
EvarDualMinimum
RlvarShiftMinimum
RlvarPrimalMinimum
EvarAnchor
RlvarAnchor
```

## Functions

```@docs
ep_jump_entropy_pooling
ep_refine_iters
ep_refine_tail_view(tv::AbstractEntropyPoolingTailView, ::VecNum)
ep_check_tail_window(::AbstractEntropyPoolingTailView, ::VecNum)
ep_diagnose_tail_view!(::JuMP.Model, ::Any, ::AbstractEntropyPoolingTailView)
ep_diagnose_views
ep_jump_views!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
