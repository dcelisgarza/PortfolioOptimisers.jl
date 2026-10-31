```@meta
Description = "L1 uncertainty sets, private API of PortfolioOptimisers.jl: expand_investable_ucs, l1_activation_ladder, l1_active_count, l1_eps_from_ladder."
```

# L1 uncertainty sets: private API

```@docs
expand_investable_ucs(set::L1UncertaintySet, imsk::BitVector, pr::AbstractPriorResult)
expand_investable_ucs(set::SignedL1UncertaintySet, imsk::BitVector, pr::AbstractPriorResult)
l1_activation_ladder
l1_active_count
l1_eps_from_ladder
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
