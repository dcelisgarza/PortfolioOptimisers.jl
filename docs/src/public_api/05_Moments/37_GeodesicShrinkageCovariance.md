```@meta
Description = "Geodesic shrinkage covariance, public API of PortfolioOptimisers.jl: AbstractCovarianceShrinkageTarget, shrinkage_target, IdentityTarget, …"
```

# Geodesic shrinkage covariance

[`GeodesicShrinkageCovariance`](@ref) moves a covariance matrix towards a target along the shortest path between two positive definite matrices. Linear shrinkage moves it along a straight line instead. A covariance estimator of your choice computes the matrix, and the shrinkage intensity sets how far it moves.

The target is built from the matrix being shrunk: the identity matrix, [`IdentityTarget`](@ref); the average variance times the identity matrix, [`ScaledIdentityTarget`](@ref); the average variance and the average covariance, [`CommonCovarianceTarget`](@ref); the variances with the average correlation, [`ConstantCorrelationTarget`](@ref); or the variances alone, [`DiagonalTarget`](@ref). A fixed positive definite matrix is a target too. A new target subtypes `AbstractCovarianceShrinkageTarget` and adds a method of `shrinkage_target`.

## Targets

```@docs
PortfolioOptimisers.AbstractCovarianceShrinkageTarget
PortfolioOptimisers.shrinkage_target
IdentityTarget
ScaledIdentityTarget
CommonCovarianceTarget
ConstantCorrelationTarget
```

## Estimator

```@docs
GeodesicShrinkageCovariance
cov(ce::GeodesicShrinkageCovariance, X::MatNum; dims::Int = 1, kwargs...)
cov(ce::GeodesicShrinkageCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
PortfolioOptimisers.partial_fit!(ce::GeodesicShrinkageCovariance, X::MatNum; dims::Int = 1, kwargs...)
cov(ce::GeodesicShrinkageCovariance; kwargs...)
port_opt_view(ce::GeodesicShrinkageCovariance, i, args...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
