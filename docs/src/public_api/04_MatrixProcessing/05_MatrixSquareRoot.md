```@meta
Description = "Matrix square root, public API of PortfolioOptimisers.jl: AbstractMatrixSquareRootAlgorithm, RidgeCholeskySquareRoot, EigenFallbackSquareRoot, …"
```

# [Matrix square root](@id api-matrix-square-root)

A square root ``\mathbf{L}`` of a covariance matrix ``\mathbf{\Sigma}`` satisfies ``\mathbf{L} \mathbf{L}^{\intercal} = \mathbf{\Sigma}``. A factor prior states its covariance through such a square root, and a second-order cone reads it. A positive definite matrix has a lower Cholesky factor. A singular or a nearly singular matrix has none, and the algorithms below state what to do then:

- `nothing` takes the plain Cholesky factor, and raises a `PosDefException`.
- [`RidgeCholeskySquareRoot`](@ref) adds a small ridge to the diagonal, and grows it until the Cholesky factor exists.
- [`EigenFallbackSquareRoot`](@ref) takes the square root of the eigendecomposition of a positive semidefinite matrix, with no ridge.

Each estimator that takes a square root of a covariance holds the algorithm in its field `mtx_sqrt`: [`FactorPrior`](@ref), [`CrossSectionalFactorPrior`](@ref), [`UncertaintySetVariance`](@ref), [`ArithmeticReturn`](@ref), [`RelaxedRiskBudgeting`](@ref), [`Kurtosis`](@ref), [`NegativeSkewness`](@ref) and [`NormBallUncertaintySetAlgorithm`](@ref). The default of each is [`EigenFallbackSquareRoot`](@ref): the plain Cholesky factor of a positive definite matrix, and the square root of the eigendecomposition of a singular positive semidefinite one. [`RidgeCholeskySquareRoot`](@ref) and `nothing`, the plain Cholesky factor that refuses a matrix that is not positive definite, are one keyword away.

```@docs
AbstractMatrixSquareRootAlgorithm
RidgeCholeskySquareRoot
EigenFallbackSquareRoot
matrix_square_root
```
