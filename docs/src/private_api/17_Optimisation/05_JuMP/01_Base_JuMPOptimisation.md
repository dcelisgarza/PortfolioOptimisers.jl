```@meta
Description = "Base JuMP Optimisation, private API of PortfolioOptimisers.jl: BaseJuMPOptimisationEstimator, JuMPOptimisationEstimator, RiskJuMPOptimisationEstimator, …"
```

# Base JuMP Optimisation: private API

```@docs
BaseJuMPOptimisationEstimator
JuMPOptimisationEstimator
RiskJuMPOptimisationEstimator
ObjectiveFunction
JuMPReturnsEstimator
JuMPConstraintEstimator
JuMPConstr_VecJuMPConstr
JuMPObj_VecJuMPObj
VecJuMPOptSol
JuMPOptSol_VecJuMPOptSol
BaseJuMPOptimisationResult
RiskJuMPOptimisationResult
NonRiskJuMPOptimisationResult
RJR_NRJR
AbstractDecompositionContract
SHARED_STATE
WeightsFromParts
PartsBoundWeights
is_time_dependent(opt::JuMPOptimisationEstimator)
reset_time_dependent_estimator(opt::JuMPOptimisationEstimator)
process_model
optimise_JuMP_model!
set_model_scales!
set_model_observations!
set_initial_w!
set_w!
set_portfolio_returns!
set_net_portfolio_returns!
set_asset_returns_plus_one!
set_asset_neg_returns_plus_one!
set_portfolio_drawdowns_plus_one!
set_risk_constraints!
has_Xap1
get_ret
get_net_X
get_ddap1
get_objective_scale
get_T
set_unit_budget!
is_unit_budget
effective_k
set_decomposition_contract!
decomposition_contract
get_Xap1
has_ddap1
has_net_X
get_X
get_risk
get_dd
has_X
has_dd
assert_shared_state
shared_set!
shared_has
shared_get
frontier_point_count
frontier_sweep_points
assert_frontier_sweep_cap
frontier_axis
set_ret_frontier_parameters!
set_risk_frontier_parameters!
frontier_sweep_axes
set_frontier_point!
frontier_sweep!
state_key
assert_state_key_free
state_set!
state_has
state_get
state_build!
mark_state!
nested_prefix
nested_index
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
