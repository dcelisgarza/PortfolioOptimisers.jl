# Near optimal centering

```@docs
NearOptimalCenteringAlgorithm
ConstrainedNearOptimalCentering
UnconstrainedNearOptimalCentering
NearOptimalCenteringResult
factory(res::NearOptimalCenteringResult, fb::Option{<:OptE_Opt})
Base.getproperty(r::NearOptimalCenteringResult, sym::Symbol)
NearOptimalSetup
NearOptimalCentering
near_optimal_centering_td_defaults
needs_previous_weights(opt::NearOptimalCentering)
factory(noc::NearOptimalCentering, w::AbstractVector)
port_opt_view(noc::NearOptimalCentering, i, X::MatNum, args...)
near_optimal_centering_risks
near_optimal_centering_setup
frontier_return_terms
return_term_ends
set_near_optimal_centering_constraints!
set_near_optimal_objective_function!
solve_noc!
assemble_near_optimal_centering_model!(::UnconstrainedNearOptimalCentering, model::JuMP.Model, noc::NearOptimalCentering, setup::NearOptimalSetup, rd::ReturnsResult)
solve_near_optimal_centering!(::UnconstrainedNearOptimalCentering, model::JuMP.Model, noc::NearOptimalCentering, setup::NearOptimalSetup)
set_noc_anchor_parameters!
set_noc_anchor!
get_overall_retcode
compute_ret_lbs(ret_frontier::VecPair, ::Nothing)
compute_ret_lbs(ret_frontier::VecPair, rt_ends::VecPair)
compute_risk_ubs(model::JuMP.Model, noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ConstrainedNearOptimalCentering}, pr::AbstractPriorResult, fees::Option{<:Fees}, w_min::VecNum, w_max::VecNum, args...)
optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                            <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
