```@meta
Description = "Matrix processing, public API of PortfolioOptimisers.jl: AbstractMatrixProcessingEstimator, AbstractMatrixProcessingAlgorithm, MatrixProcessing, …"
```

# Matrix processing

A matrix processing estimator changes a covariance or correlation matrix after it is estimated. It has four steps:

 1. Positive definite projection.
 2. Denoising.
 3. Detoning.
 4. A custom step that you write.

Run the positive definite projection first. The other steps work best on a positive definite matrix, and the denoising and detoning steps can project the matrix again.

For the other steps, the only order with a known reason is to denoise before you detone. You set the order as a tuple or vector of the step symbols `:pdm`, `:dn`, `:dt` and `:alg`, and the steps run from left to right. The default order is `(:pdm, :dn, :dt, :alg)`.

```@docs
AbstractMatrixProcessingEstimator
AbstractMatrixProcessingAlgorithm
MatrixProcessing
matrix_processing!
matrix_processing
matrix_processing_algorithm!(::Nothing, sigma::MatNum, args...; kwargs...)
matrix_processing_algorithm(::Nothing, sigma::MatNum, args...; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
