# Hierarchical Risk Parity

```@docs
HierarchicalRiskParity
needs_previous_weights(opt::HierarchicalRiskParity)
is_time_dependent(opt::HierarchicalRiskParity)
update_time_dependent_estimator(opt::HierarchicalRiskParity, ctx::TimeDependentContext)
reset_time_dependent_estimator(opt::HierarchicalRiskParity)
port_opt_view(hrp::HierarchicalRiskParity, i, X::MatNum, args...)
split_factor_weight_constraints(alpha::Number, wb::WeightBounds, w::VecNum, lc::VecNum, rc::VecNum)
hrp_scalarised_risk
optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
```
