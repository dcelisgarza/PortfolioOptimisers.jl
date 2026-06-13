# Hierarchical Risk Parity

```@docs
HierarchicalRiskParity
needs_previous_weights(opt::HierarchicalRiskParity)
port_opt_view(hrp::HierarchicalRiskParity, i, X::MatNum)
split_factor_weight_constraints(alpha::Number, wb::WeightBounds, w::VecNum, lc::VecNum, rc::VecNum)
hrp_scalarised_risk(::SumScalariser, wu::MatNum, wk::VecNum, rku::VecNum, lc::VecNum, rc::VecNum, rs::VecOptRM, X::MatNum, fees::Option{<:Fees})
optimise(hrp::HierarchicalRiskParity{<:Any, <:Any, <:Any, <:Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1, kwargs...)
```
