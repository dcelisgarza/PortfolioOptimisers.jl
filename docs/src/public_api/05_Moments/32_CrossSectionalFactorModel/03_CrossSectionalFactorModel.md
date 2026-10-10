```@meta
Description = "Cross-Sectional Factor Model, public API of PortfolioOptimisers.jl: CrossSectionalFactorModel, FactorDiagnosticResult, port_opt_view, regression, …"
```

# [Cross-Sectional Factor Model](@id api-cross-sectional-factor-model)

## Types

```@docs
CrossSectionalFactorModel
FactorDiagnosticResult
```

## Functions

```@docs
port_opt_view(csfm::CrossSectionalFactorModel, i, args...)
regression(csfm::CrossSectionalFactorModel, args...)
cross_sectional_factor_returns
port_opt_view(r::FactorDiagnosticResult, i, args...)
cs_diagnostic_factor_names
```
