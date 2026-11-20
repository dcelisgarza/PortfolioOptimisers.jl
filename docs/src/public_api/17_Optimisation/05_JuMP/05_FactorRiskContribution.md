```@meta
Description = "Factor risk contribution, public API of PortfolioOptimisers.jl: FactorRiskContributionResult, FactorRiskContribution, factory, port_opt_view, _optimise, …"
```

# Factor risk contribution

```@docs
FactorRiskContributionResult
FactorRiskContribution
factory(res::FactorRiskContributionResult, fb::Option{<:OptE_Opt_FbChain})
port_opt_view(frc::FactorRiskContribution, i, X::MatNum, args...)
_optimise(frc::FactorRiskContribution, rd::ReturnsResult = ReturnsResult())
needs_previous_weights(opt::FactorRiskContribution)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
