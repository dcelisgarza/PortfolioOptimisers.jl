# Matrix processing

Co-moment matrices can be post-processed after being computed. These processes are often complementary but there is no set order.

At the base level, we have four possible post processing steps:

 1. Positive definite projection.
 2. Denoising.
 3. Detoning.
 4. Custom process.

The only set order is that positive definite projection should come first. This is because the other post-processing methods work best with positive definite matrices, and may use positive definite projection internally.

Aside from this, there is no set canonical order, the closest to a heuristic we can justify is to denoise before detoning. As such we provide all permutations of ordering for the last three steps.

```@docs
AbstractMatrixProcessingEstimator
AbstractMatrixProcessingAlgorithm
AbstractMatrixProcessingOrder
DenoiseDetoneAlg
DenoiseAlgDetone
DetoneDenoiseAlg
DetoneAlgDenoise
AlgDenoiseDetone
AlgDetoneDenoise
DenoiseDetoneAlgMatrixProcessing
matrix_processing!
matrix_processing
matrix_processing_algorithm!(::Nothing, sigma::MatNum, args...; kwargs...)
matrix_processing_algorithm(::Nothing, sigma::MatNum, args...; kwargs...)
```
