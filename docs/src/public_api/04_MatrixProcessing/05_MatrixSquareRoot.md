```@meta
Description = "Matrix square root, public API of PortfolioOptimisers.jl: AbstractMatrixSquareRootAlgorithm, RidgeCholeskySquareRoot, EigenFallbackSquareRoot, …"
```

# [Matrix square root](@id api-matrix-square-root)

A square root ``\mathbf{L}`` of a covariance matrix ``\mathbf{\Sigma}`` satisfies ``\mathbf{L} \mathbf{L}^{\intercal} = \mathbf{\Sigma}``. A factor prior states its covariance through such a square root, and a second-order cone reads it. A positive definite matrix has a lower Cholesky factor. A singular or a nearly singular matrix has none, and the algorithms below state what to do then:

- `nothing` takes the plain Cholesky factor, and raises a `PosDefException`.
- [`RidgeCholeskySquareRoot`](@ref) adds a small ridge to the diagonal, and grows it until the Cholesky factor exists.
- [`EigenFallbackSquareRoot`](@ref) takes the square root of the eigendecomposition of a positive semidefinite matrix, with no ridge.

`FactorPrior` and `CrossSectionalFactorPrior` take the algorithm in their field `sqrt_alg`.

```@docs
AbstractMatrixSquareRootAlgorithm
RidgeCholeskySquareRoot
EigenFallbackSquareRoot
matrix_square_root
```
