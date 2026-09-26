```@meta
Description = "Denoise, private API of PortfolioOptimisers.jl: find_max_eval, _denoise!."
```

# Denoise: private API

[`denoise!`](@ref) and [`denoise`](@ref) call the functions below. [`find_max_eval`](@ref) estimates the upper edge of the Marčenko-Pastur distribution, which separates the eigenvalues of noise from those of signal. A new denoising algorithm adds a method to `_denoise!`.

```@docs
find_max_eval
_denoise!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
