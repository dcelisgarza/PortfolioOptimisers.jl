```@meta
Description = "EW beta descriptors, public API of PortfolioOptimisers.jl: EWBeta, EWMacroSensitivity, EWDownsideBeta, descriptor, EWMarketBeta."
```

# [EW beta descriptors](@id api-ew-beta-descriptors)

## Types

```@docs
EWBeta
EWMacroSensitivity
EWDownsideBeta
```

## Functions

```@docs
descriptor(de::EWBeta, rd::ReturnsResult)
descriptor(de::EWMacroSensitivity, rd::ReturnsResult)
descriptor(de::EWDownsideBeta, rd::ReturnsResult)
EWMarketBeta
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
