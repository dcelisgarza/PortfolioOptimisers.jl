```@meta
Description = "Subset resampling, public API of PortfolioOptimisers.jl: SubsetResamplingResult, SubsetResampling, factory, port_opt_view, optimise, needs_previous_weights."
```

# Subset resampling

```@docs
SubsetResamplingResult
SubsetResampling
factory(sr::SubsetResamplingResult, fb::Option{<:OptE_Opt_FbChain})
factory(sr::SubsetResampling, w::AbstractVector)
port_opt_view(sr::SubsetResampling, i, X::MatNum, args...)
optimise(sr::SubsetResampling{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, branchorder::Symbol = :optimal, str_names::Bool = false, save::Bool = true, kwargs...)
needs_previous_weights(opt::SubsetResampling)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
