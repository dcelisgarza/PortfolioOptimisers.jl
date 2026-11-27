```@meta
Description = "Spectral Seriation, public API of PortfolioOptimisers.jl: SpectralSeriationState, port_opt_view, SpectralSeriation, SpectralSeriationResult, seriate, …"
```

# Spectral Seriation

```@docs
SpectralSeriationState
port_opt_view(x::SpectralSeriationState, i, args...)
SpectralSeriation
SpectralSeriationResult
port_opt_view(ser::SpectralSeriationResult, i, args...)
seriate(ser::SpectralSeriation, X::MatNum; dims::Int = 1, kwargs...)
partial_fit!(ser::SpectralSeriation, X::MatNum; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
