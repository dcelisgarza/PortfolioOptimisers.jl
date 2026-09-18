```@meta
Description = "Factor risk contribution, public API of PortfolioOptimisers.jl: FactorRiskContributionResult, FactorRiskContribution, factory, Base.getproperty, …"
```

# Factor risk contribution

```@docs
FactorRiskContributionResult
FactorRiskContribution
factory(res::FactorRiskContributionResult, fb::Option{<:OptE_Opt_FbChain})
Base.getproperty(r::FactorRiskContributionResult, sym::Symbol)
port_opt_view(frc::FactorRiskContribution, i, X::MatNum, args...)
optimise(frc::FactorRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, rd::ReturnsResult; dims::Int = 1, str_names::Bool = false, save::Bool = true, kwargs...)
needs_previous_weights(opt::FactorRiskContribution)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
