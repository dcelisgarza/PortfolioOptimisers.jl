# Risk budgeting

```@docs
RiskBudgetingResult
factory(res::RiskBudgetingResult, fb::Option{<:OptE_Opt})
risk_budgeting_td_defaults
Base.getproperty(r::RiskBudgetingResult, sym::Symbol)
ProcessedRiskBudgetingAttributes
ProcessedFactorRiskBudgetingAttributes
ProcessedAssetRiskBudgetingAttributes
RiskBudgetingFormulation
port_opt_view(::RiskBudgetingFormulation, ::Any, args...)
port_opt_view(alg::LogRiskBudgeting{Nothing}, i, args...)
port_opt_view(alg::LogRiskBudgeting{<:VecInt}, i, args...)
LogRiskBudgeting
MixedIntegerRiskBudgeting
RiskBudgetingAlgorithm
AssetRiskBudgeting
FactorRiskBudgeting
RiskBudgeting
needs_previous_weights(opt::RiskBudgeting)
factory(rb::RiskBudgeting, w::AbstractVector)
port_opt_view(rb::RiskBudgeting, i, X::MatNum, args...)
_set_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting, w::VecJuMPScalar; strict::Bool = false)
set_risk_budgeting_constraints!
set_rb_mip_w!
optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
```
