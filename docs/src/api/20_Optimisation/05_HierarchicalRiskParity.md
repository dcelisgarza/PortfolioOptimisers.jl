# Hierarchical Risk Parity

```@docs
HierarchicalRiskParity
hierarchical_risk_parity_td_defaults
needs_previous_weights(opt::HierarchicalRiskParity)
port_opt_view(hrp::HierarchicalRiskParity, i, X::MatNum, args...)
split_factor_weight_constraints(alpha::Number, wb::WeightBounds, w::VecNum, lc::VecNum, rc::VecNum)
hrp_scalarised_risk
optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
```
