```@meta
Description = "Matrix processing, private API of PortfolioOptimisers.jl: matrix_processing_block!, assert_finite_block, matrix_processing_step!, …"
```

# Matrix processing: private API

These are the helpers of [`matrix_processing!`](@ref). `matrix_processing_block!` runs the steps on the part of the matrix that holds no `NaN`, so the entries of an asset that the fit left out stay `NaN`. `matrix_processing_step!` applies one named step, such as the positive definite projection or the denoising.

```@docs
matrix_processing_block!
PortfolioOptimisers.assert_finite_block
matrix_processing_step!
PortfolioOptimisers.assert_shape_only_matrix_processing
```
