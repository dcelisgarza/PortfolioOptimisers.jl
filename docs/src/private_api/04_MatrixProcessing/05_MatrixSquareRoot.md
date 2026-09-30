```@meta
Description = "Matrix square root, private API of PortfolioOptimisers.jl: ridge_cholesky."
```

# Matrix square root: private API

`ridge_cholesky` is the ridge schedule of [`RidgeCholeskySquareRoot`](@ref). It returns `nothing` where no ridge makes the matrix factorise, so a caller that must not raise, such as the Mahalanobis regime statistic, reads it directly.

```@docs
PortfolioOptimisers.ridge_cholesky
```
