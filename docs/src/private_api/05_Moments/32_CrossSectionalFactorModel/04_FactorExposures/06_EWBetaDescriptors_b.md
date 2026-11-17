```@meta
Description = "EW beta descriptors (b), private API of PortfolioOptimisers.jl: EWMacroSensitivityState, EWDownsideBetaState, CarriedReference, Base.copy, ew_macro_beta, …"
```

# EW beta descriptors (b): private API

## Types

```@docs
EWMacroSensitivityState
EWDownsideBetaState
CarriedReference
```

## Functions

```@docs
Base.copy(x::PortfolioOptimisers.EWMacroSensitivityState)
ew_macro_beta
ew_macro_slope
ew_macro_partial
ew_macro_reference
ew_macro_windows
ew_macro_refuse
Base.copy(x::PortfolioOptimisers.EWDownsideBetaState)
ew_downside_beta_state
ew_downside_beta_series!
ew_downside_beta_step!
ew_macro_state
ew_macro_carried
ew_beta_fold
show_fields(de::Union{PortfolioOptimisers.EWBeta, PortfolioOptimisers.EWMacroSensitivity, PortfolioOptimisers.EWDownsideBeta})
```
