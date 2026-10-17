```@meta
Description = "EW beta descriptors (b), private API of PortfolioOptimisers.jl: EWMacroSensitivityState, EWDownsideBetaState, Base.copy, ew_macro_sensitivity_state, …"
```

# EW beta descriptors (b): private API

## Types

```@docs
EWMacroSensitivityState
EWDownsideBetaState
```

## Functions

```@docs
Base.copy(x::PortfolioOptimisers.EWMacroSensitivityState)
ew_macro_sensitivity_state
ew_macro_sensitivity_series!
ew_macro_reference
Base.copy(x::PortfolioOptimisers.EWDownsideBetaState)
ew_downside_beta_state
ew_downside_beta_series!
ew_block_state
ew_macro_carried
ew_beta_fold
show_fields(de::Union{PortfolioOptimisers.EWBeta, PortfolioOptimisers.EWMacroSensitivity, PortfolioOptimisers.EWDownsideBeta})
```
