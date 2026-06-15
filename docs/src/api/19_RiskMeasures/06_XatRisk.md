# X at Risk

```@docs
ValueatRiskFormulation
factory(alg::ValueatRiskFormulation, args...; kwargs...)
port_opt_view(r::ValueatRiskFormulation, ::Any, args...)
port_opt_view(alg::DistributionValueatRisk, i, args...)
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
