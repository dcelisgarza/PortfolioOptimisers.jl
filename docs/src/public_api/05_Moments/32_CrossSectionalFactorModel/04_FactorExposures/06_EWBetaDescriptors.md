```@meta
Description = "Exponentially Weighted Beta Descriptors, public API of PortfolioOptimisers.jl: EWBeta, EWMacroSensitivity, EWDownsideBeta, descriptor, EWMarketBeta."
```

# [Exponentially Weighted Beta Descriptors](@id api-ew-beta-descriptors)

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
