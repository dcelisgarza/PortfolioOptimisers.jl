```@meta
Description = "Nested Clustered, public API of PortfolioOptimisers.jl: NestedClusteredResult, NestedClustered, factory, port_opt_view, optimise, needs_previous_weights."
```

# Nested Clustered

```@docs
NestedClusteredResult
NestedClustered
factory(res::NestedClusteredResult, fb::Option{<:OptE_Opt_FbChain})
factory(nco::NestedClustered, w::AbstractVector)
port_opt_view(nco::NestedClustered, i, X::MatNum, args...)
optimise(nco::NestedClustered{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false, save::Bool = true, kwargs...)
needs_previous_weights(opt::NestedClustered)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
