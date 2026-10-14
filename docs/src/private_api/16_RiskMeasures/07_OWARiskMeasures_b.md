```@meta
Description = "OWA risk measures (b), private API of PortfolioOptimisers.jl: odd_even_merge_pass!, bitonic_sort!, bitonic_merge!, assert_owa_formulation_weights, …"
```

# OWA risk measures (b): private API

```@docs
odd_even_merge_pass!
bitonic_sort!
bitonic_merge!
assert_owa_formulation_weights(::SortingNetworkOrderedWeightsArray, w::VecNum)
resolve_deferred_quantities(x::ComposedFunction{typeof(reverse), <:AbstractOrderedWeightsArrayFunction}, pr::AbstractPriorResult)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
