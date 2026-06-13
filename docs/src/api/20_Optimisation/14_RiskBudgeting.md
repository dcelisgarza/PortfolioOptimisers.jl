# Risk budgeting

```@docs
RiskBudgetingResult
factory(res::RiskBudgetingResult, fb::Option{<:OptE_Opt})
Base.getproperty(r::RiskBudgetingResult, sym::Symbol)
ProcessedFactorRiskBudgetingAttributes
ProcessedAssetRiskBudgetingAttributes
RiskBudgetingFormulation
port_opt_view(::RiskBudgetingFormulation, args...)
port_opt_view(alg::LogRiskBudgeting{Nothing}, i)
port_opt_view(alg::LogRiskBudgeting{<:VecInt}, i)
LogRiskBudgeting
MixedIntegerRiskBudgeting
RiskBudgetingAlgorithm
AssetRiskBudgeting
FactorRiskBudgeting
RiskBudgeting
needs_previous_weights(opt::RiskBudgeting)
factory(rb::RiskBudgeting, w::AbstractVector)
port_opt_view(rb::RiskBudgeting, i, X::MatNum)
port_opt_view(r::AssetRiskBudgeting, i)
port_opt_view(r::FactorRiskBudgeting, i)
_set_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting, w::VecJuMPScalar; strict::Bool = false)
set_risk_budgeting_constraints!
set_rb_mip_w!
optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
```
