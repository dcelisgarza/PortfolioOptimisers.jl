```@meta
Description = "Geodesic shrinkage covariance, private API of PortfolioOptimisers.jl: AbstractCovarianceShrinkageTarget, GeodesicShrinkageTarget, shrinkage_target, …"
```

# Geodesic shrinkage covariance: private API

These are the target family and the helpers of [`GeodesicShrinkageCovariance`](@ref). `shrinkage_target` builds the target matrix, `geodesic_point` computes the shrunk matrix, and `geodesic_shrinkage!` runs the repair and the shrinkage on the assets with a finite variance.

```@docs
PortfolioOptimisers.AbstractCovarianceShrinkageTarget
PortfolioOptimisers.GeodesicShrinkageTarget
PortfolioOptimisers.shrinkage_target
PortfolioOptimisers.assert_shrinkage_target
PortfolioOptimisers.geodesic_power
PortfolioOptimisers.geodesic_point
PortfolioOptimisers.geodesic_shrinkage!
PortfolioOptimisers.frame_cov2cor!
gap_fill_value(::GeodesicShrinkageCovariance)
PortfolioOptimisers.supports_partial_fit(ce::GeodesicShrinkageCovariance)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
