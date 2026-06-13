# Factor risk contribution

```@docs
FactorRiskContributionResult
factory(res::FactorRiskContributionResult, fb::Option{<:OptE_Opt})
Base.getproperty(r::FactorRiskContributionResult, sym::Symbol)
FactorRiskContribution
needs_previous_weights(opt::FactorRiskContribution)
port_opt_view(frc::FactorRiskContribution, i, X::MatNum)
set_factor_risk_contribution_constraints!(model::JuMP.Model, re::RegE_Reg, rd::ReturnsResult, flag::Bool, wi::Option{<:VecNum})
optimise(frc::FactorRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                              <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
```
