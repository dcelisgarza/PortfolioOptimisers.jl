# X at Risk

```@docs
ValueatRiskFormulation
factory(alg::ValueatRiskFormulation, args...; kwargs...)
valueat_risk_formulation_view(r::ValueatRiskFormulation, args...)
valueat_risk_formulation_view(alg::DistributionValueatRisk, i)
MIPValueatRisk
DistributionValueatRisk
factory(alg::DistributionValueatRisk, pr::AbstractPriorResult, args...; kwargs...)
ValueatRisk
factory(r::ValueatRisk, pr::AbstractPriorResult, args...; kwargs...)
ValueatRiskRange
factory(r::ValueatRiskRange, pr::AbstractPriorResult, args...; kwargs...)
DrawdownatRisk
factory(r::DrawdownatRisk, pr::AbstractPriorResult, args...; kwargs...)
RelativeDrawdownatRisk
factory(r::RelativeDrawdownatRisk, pr::AbstractPriorResult, args...; kwargs...)
CholRM
absolute_drawdown_vec
relative_drawdown_vec(x::VecNum)
```
