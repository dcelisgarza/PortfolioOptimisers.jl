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
needs_previous_weights(opt::NearOptimalCentering)
factory(noc::NearOptimalCentering, w::AbstractVector)
opt_view(noc::NearOptimalCentering, i, X::MatNum)
near_optimal_centering_risks
near_optimal_centering_setup
set_near_optimal_centering_constraints!
set_near_optimal_objective_function!
solve_noc!
get_overall_retcode
compute_ret_lbs(lbs::VecNum, args...)
compute_ret_lbs(lbs::Frontier, model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator, pr::AbstractPriorResult, fees::Option{<:Fees})
compute_ret_lbs(lbs::Frontier, rt_min::Number, rt_max::Number)
compute_risk_ubs(model::JuMP.Model, mr::MeanRisk, ret::JuMPReturnsEstimator, pr::AbstractPriorResult, fees::Option{<:Fees})
compute_risk_ubs(model::JuMP.Model, noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ConstrainedNearOptimalCentering}, pr::AbstractPriorResult, fees::Option{<:Fees}, w_min::VecNum, w_max::VecNum)
optimise(noc::NearOptimalCentering{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                            <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  str_names::Bool = false, save::Bool = true, kwargs...)
```
