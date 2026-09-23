```@meta
Description = "Denoise, public API of PortfolioOptimisers.jl: AbstractDenoiseEstimator, AbstractDenoiseAlgorithm, SpectralDenoise, FixedDenoise, ShrunkDenoise, Denoise, …"
```

# Denoise

A covariance matrix estimated from returns mixes the relations between the assets with sampling noise. For a given number of observations and assets, random matrix theory gives the range of eigenvalues that pure noise produces.

Denoising changes the small eigenvalues inside that range, so they have less effect on the result [mlp1,mpdist](@cite). It also lowers the condition number of the matrix, so a solver that uses the matrix is more stable.

```@docs
AbstractDenoiseEstimator
AbstractDenoiseAlgorithm
SpectralDenoise
FixedDenoise
ShrunkDenoise
Denoise
denoise!
denoise
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
