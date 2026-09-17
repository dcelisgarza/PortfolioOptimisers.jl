```@meta
Description = "Matrix processing, private API of PortfolioOptimisers.jl: AbstractMatrixProcessingEstimator, AbstractMatrixProcessingAlgorithm, matrix_processing_block!, …"
```

# Matrix processing: private API

The internal interface behind matrix post-processing: the `:alg` step's algorithm family, and
the block- and step-level helpers `matrix_processing!` drives.

```@docs
AbstractMatrixProcessingEstimator
AbstractMatrixProcessingAlgorithm
matrix_processing_block!
PortfolioOptimisers.assert_finite_block
matrix_processing_step!
PortfolioOptimisers.assert_shape_only_matrix_processing
matrix_processing_algorithm!(::Nothing, sigma::MatNum, args...; kwargs...)
matrix_processing_algorithm(::Nothing, sigma::MatNum, args...; kwargs...)
```
