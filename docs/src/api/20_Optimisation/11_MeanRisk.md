# Mean Risk

```@docs
MeanRiskResult
mean_risk_td_defaults
factory(res::MeanRiskResult, fb::Option{<:OptE_Opt})
Base.getproperty(r::MeanRiskResult, sym::Symbol)
MeanRisk
needs_previous_weights(opt::MeanRisk)
factory(mr::MeanRisk, w::AbstractVector)
port_opt_view(mr::MeanRisk, i, X::MatNum, args...)
solve_mean_risk!
return_term
compute_ret_lbs
_rebuild_risk_frontier
rebuild_risk_frontier
unresolved_risk_frontier
risk_frontier_owners
compute_risk_ubs
optimise(mr::MeanRisk{<:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
