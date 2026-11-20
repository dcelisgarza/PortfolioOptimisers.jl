```@meta
Description = "Subset resampling, public API of PortfolioOptimisers.jl: SubsetResamplingResult, PowerSubsetSize, SubsetResampling, factory, port_opt_view, _optimise, …"
```

# Subset resampling

```@docs
SubsetResamplingResult
PowerSubsetSize
SubsetResampling
factory(sr::SubsetResamplingResult, fb::Option{<:OptE_Opt_FbChain})
factory(sr::SubsetResampling, w::AbstractVector)
port_opt_view(sr::SubsetResampling, i, X::MatNum, args...)
_optimise(sr::SubsetResampling, rd::ReturnsResult)
needs_previous_weights(opt::SubsetResampling)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
