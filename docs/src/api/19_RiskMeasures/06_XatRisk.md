# X at Risk

```@docs
ValueatRiskFormulation
factory(alg::ValueatRiskFormulation, args...; kwargs...)
port_opt_view(r::ValueatRiskFormulation, ::Any, args...)
compute_value_at_risk_z
compute_value_at_risk_cz
port_opt_view(alg::DistributionValueatRisk, i, args...)
MIPValueatRisk
DistributionValueatRisk
factory(alg::DistributionValueatRisk, pr::AbstractPriorResult, args...; kwargs...)
ValueatRisk
resolve_deferred_quantities(x::ValueatRisk, pr::AbstractPriorResult)
ValueatRiskRange
resolve_deferred_quantities(x::ValueatRiskRange, pr::AbstractPriorResult)
DrawdownatRisk
resolve_deferred_quantities(x::DrawdownatRisk, pr::AbstractPriorResult)
RelativeDrawdownatRisk
resolve_deferred_quantities(x::RelativeDrawdownatRisk,
                            pr::AbstractPriorResult)
CholRM
absolute_drawdown_vec
relative_drawdown_vec(x::VecNum)
drawdown_at_risk
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
