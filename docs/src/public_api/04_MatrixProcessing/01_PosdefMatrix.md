```@meta
Description = "Positive definite matrix projection, public API of PortfolioOptimisers.jl: AbstractPosdefEstimator, Posdef, posdef, posdef!."
```

# [Positive definite matrix projection](@id api-positive-definite-matrix-projection)

Many optimisations need a positive definite covariance or correlation matrix to have a unique solution. An estimated matrix can fail this for three reasons: collinear assets, fewer observations than assets, and floating-point rounding. Such a matrix has zero or negative eigenvalues.

- A zero eigenvalue means that a combination of the assets has no variance. A linear or quadratic system with the matrix then has no unique solution.
- A negative eigenvalue comes from rounding in a badly conditioned matrix, usually one with highly collinear assets, or from an estimator that does not guarantee a positive semi-definite result.

The types and functions below replace such a matrix with the nearest positive definite matrix. The result changes the matrix as little as possible, and it has no zero or negative eigenvalue.

```@docs
AbstractPosdefEstimator
Posdef
posdef
posdef!
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
