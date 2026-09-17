```@meta
Description = "Expected Risk, public API of PortfolioOptimisers.jl: expected_risk, expected_risk_from_returns, number_effective_assets, risk_contribution, …"
```

# Expected Risk

```@docs
expected_risk
expected_risk_from_returns
expected_risk_from_returns(r::AbstractBaseRiskMeasure, X::VecNum; kwargs...)
expected_risk_from_returns(r::AbstractBaseRiskMeasure, X::VecVecNum; kwargs...)
number_effective_assets
risk_contribution
factor_risk_contribution
rolling_window_measure
```
