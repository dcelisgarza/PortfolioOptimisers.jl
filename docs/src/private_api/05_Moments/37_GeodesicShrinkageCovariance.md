```@meta
Description = "Geodesic shrinkage covariance, private API of PortfolioOptimisers.jl: GeodesicShrinkageTarget, assert_shrinkage_target, geodesic_power, geodesic_point, …"
```

# Geodesic shrinkage covariance: private API

These are the helpers of [`GeodesicShrinkageCovariance`](@ref). `geodesic_point` computes the shrunk matrix, and `geodesic_shrinkage!` runs the repair and the shrinkage on the assets with a finite variance.

```@docs
PortfolioOptimisers.GeodesicShrinkageTarget
PortfolioOptimisers.assert_shrinkage_target
PortfolioOptimisers.geodesic_power
PortfolioOptimisers.geodesic_point
PortfolioOptimisers.geodesic_shrinkage!
PortfolioOptimisers.frame_cov2cor!
gap_fill_value(::GeodesicShrinkageCovariance)
PortfolioOptimisers.supports_partial_fit(ce::GeodesicShrinkageCovariance)
```
