# Denoise

Real world data in general is often noisy. Financial data, is subject to myriad sources of noise, from latency, to arbitrage, to price uncertainty, to inherent randomness.

Denoising is about reducing or removing noise from signal. This can be done by modifying the small eigenvalues associated with noise to reduce their impact on the result [mlp1,mpdist](@cite). Denoising also reduces the condition number, thus improving the numerical behaviour of the final matrix.

```@docs
AbstractDenoiseEstimator
AbstractDenoiseAlgorithm
SpectralDenoise
FixedDenoise
ShrunkDenoise
Denoise
errPDF
find_max_eval
_denoise!
denoise!
denoise
```
