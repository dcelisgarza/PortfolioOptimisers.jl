# Positive definite matrix projection

All co-moment matrices are supposed to be positive definite. However, non-positive definite matrices can arise due to collinearity, having fewer observations than observables, and floating point innacuracies. Non-positive definite matrices have zero or negative eigenvalues.

  - Zero eigenvalues imply collinearity between assets, which means the system is underdetermined and therefore a linear or quadratic system involving the matrix does not have a unique solution.
  - Negative eigenvalues imply numerical stability issues, which usually arise from poorly conditioned systems, which for covariance and correlation matrices arise from high collinearity.

In order to obtain unique results and improve numerical stability, these non-positive definite matrices can be projected to the nearest positive definite matrix. This keeps the underlying relationships as intact as possible, while ensuring they have the proper numerical characteristics.

These types and functions let us do so.

```@docs
AbstractPosdefEstimator
Posdef
posdef!
posdef
```
