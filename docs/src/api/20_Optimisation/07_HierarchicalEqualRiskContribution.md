# Hierarchical Equal Risk Contribution

```@docs
HierarchicalEqualRiskContribution
needs_previous_weights(opt::HierarchicalEqualRiskContribution)
factory(hec::HierarchicalEqualRiskContribution, w::AbstractVector)
port_opt_view(hec::HierarchicalEqualRiskContribution, i, X::MatNum, args...)
herc_scalarised_risk_o!
herc_scalarised_risk_i!
herc_risk
optimise(hec::HierarchicalEqualRiskContribution{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing},
                  rd::ReturnsResult = ReturnsResult(); dims::Int = 1,
                  branchorder::Symbol = :optimal, kwargs...)
```
