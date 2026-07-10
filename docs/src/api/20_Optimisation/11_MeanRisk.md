# Mean Risk

```@docs
MeanRiskResult
factory(res::MeanRiskResult, fb::Option{<:OptE_Opt})
Base.getproperty(r::MeanRiskResult, sym::Symbol)
MeanRisk
needs_previous_weights(opt::MeanRisk)
is_time_dependent(opt::MeanRisk)
update_time_dependent_estimator(opt::MeanRisk, ctx::TimeDependentContext)
factory(mr::MeanRisk, w::AbstractVector)
port_opt_view(mr::MeanRisk, i, X::MatNum, args...)
solve_mean_risk!
compute_ret_lbs
_rebuild_risk_frontier
rebuild_risk_frontier
compute_risk_ubs
optimise(mr::MeanRisk{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
```
