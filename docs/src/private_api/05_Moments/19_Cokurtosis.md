```@meta
Description = "Cokurtosis, private API of PortfolioOptimisers.jl: CokurtosisEstimator, coverage_cokurtosis, _cokurtosis."
```

# Cokurtosis: private API

```@docs
CokurtosisEstimator
coverage_cokurtosis
coverage_cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:FullMoment}, ::Nothing, X::MatNum)
coverage_cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:SemiMoment}, ::Nothing, X::MatNum)
coverage_cokurtosis(kte::Cokurtosis, cvg::CoveragePolicy, X::MatNum)
_cokurtosis
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
