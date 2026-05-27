# Risk budgeting

```@docs
RiskBudgetingResult
factory(res::RiskBudgetingResult, fb::Option{<:OptE_Opt})
Base.getproperty(r::RiskBudgetingResult, sym::Symbol)
ProcessedFactorRiskBudgetingAttributes
ProcessedAssetRiskBudgetingAttributes
RiskBudgetingFormulation
risk_budgeting_formulation_view(::RiskBudgetingFormulation, args...)
risk_budgeting_formulation_view(alg::LogRiskBudgeting{Nothing}, i)
risk_budgeting_formulation_view(alg::LogRiskBudgeting{<:VecInt}, i)
LogRiskBudgeting
MixedIntegerRiskBudgeting
RiskBudgetingAlgorithm
AssetRiskBudgeting
FactorRiskBudgeting
RiskBudgeting
needs_previous_weights(opt::RiskBudgeting)
factory(rb::RiskBudgeting, w::AbstractVector)
opt_view(rb::RiskBudgeting, i, X::MatNum)
risk_budgeting_algorithm_view(r::AssetRiskBudgeting, i)
risk_budgeting_algorithm_view(r::FactorRiskBudgeting, i)
_set_risk_budgeting_constraints!(model::JuMP.Model, rb::RiskBudgeting, w::VecJuMPScalar; strict::Bool = false)
set_risk_budgeting_constraints!
set_rb_mip_w!
optimise(rb::RiskBudgeting{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
```
