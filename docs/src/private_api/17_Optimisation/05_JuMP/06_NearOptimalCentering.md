```@meta
Description = "Near optimal centering, private API of PortfolioOptimisers.jl: NearOptimalCenteringAlgorithm, NearOptimalSetup, near_optimal_centering_td_defaults, …"
```

# Near optimal centering: private API

```@docs
NearOptimalCenteringAlgorithm
NearOptimalSetup
near_optimal_centering_td_defaults
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
```
