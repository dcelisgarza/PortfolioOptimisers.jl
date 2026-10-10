```@meta
Description = "EW beta descriptors (b), public API of PortfolioOptimisers.jl: EWMacroSensitivity, EWDownsideBeta, descriptor, partial_fit!, merge_states."
```

# EW beta descriptors (b)

## Types

```@docs
EWMacroSensitivity
EWDownsideBeta
```

## Functions

```@docs
descriptor(de::EWMacroSensitivity, rd::ReturnsResult)
descriptor(de::EWDownsideBeta, rd::ReturnsResult)
PortfolioOptimisers.partial_fit!(de::Union{EWBeta, EWMacroSensitivity, EWDownsideBeta}, rd::ReturnsResult)
merge_states(::PortfolioOptimisers.EWMacroSensitivityState, ::PortfolioOptimisers.EWMacroSensitivityState)
merge_states(::PortfolioOptimisers.EWDownsideBetaState, ::PortfolioOptimisers.EWDownsideBetaState)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
