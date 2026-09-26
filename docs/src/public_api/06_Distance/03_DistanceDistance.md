```@meta
Description = "Distances of Distances, public API of PortfolioOptimisers.jl: DistanceDistance, distance, cor_and_dist."
```

# [Distances of Distances](@id api-distances-of-distances)

```@docs
DistanceDistance
distance(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
distance(de::DistanceDistance, rho::MatNum, args...; kwargs...)
cor_and_dist(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
