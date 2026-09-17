```@meta
Description = "Coskewness, private API of PortfolioOptimisers.jl: negative_spectral_coskewness, negative_spectral_part, coverage_coskewness, _coskewness."
```

# Coskewness: private API

```@docs
negative_spectral_coskewness
negative_spectral_part
coverage_coskewness
coverage_coskewness(ske::Coskewness{<:Any, <:Any, <:FullMoment}, ::Nothing, X::MatNum)
coverage_coskewness(ske::Coskewness{<:Any, <:Any, <:SemiMoment}, ::Nothing, X::MatNum)
coverage_coskewness(ske::Coskewness, cvg::CoveragePolicy, X::MatNum)
_coskewness
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
