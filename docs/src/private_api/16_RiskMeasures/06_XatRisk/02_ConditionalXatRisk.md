```@meta
Description = "Conditional X at Risk, private API of PortfolioOptimisers.jl: RMCVaR, RMCVaRRg, RMCDaR, resolve_deferred_quantities, conditional_drawdown_at_risk."
```

# Conditional X at Risk: private API

```@docs
RMCVaR
RMCVaRRg
RMCDaR
resolve_deferred_quantities(x::DistributionallyRobustConditionalValueatRisk, pr::AbstractPriorResult)
resolve_deferred_quantities(x::DistributionallyRobustConditionalValueatRiskRange, pr::AbstractPriorResult)
resolve_deferred_quantities(x::DistributionallyRobustConditionalDrawdownatRisk, pr::AbstractPriorResult)
conditional_drawdown_at_risk
```
