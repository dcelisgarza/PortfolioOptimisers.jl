```@meta
Description = "OWA Risk Measure Constraints, private API of PortfolioOptimisers.jl: set_owa_constraints!, set_risk_constraints!."
```

# OWA Risk Measure Constraints: private API

```@docs
set_owa_constraints!
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArray{<:Any, <:Any, <:ExactOrderedWeightsArray}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ExactOrderedWeightsArray}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArray{<:Any, <:Any, <:ApproxOrderedWeightsArray}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::OrderedWeightsArrayRange{<:Any, <:Any, <:Any, <:ApproxOrderedWeightsArray}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
```
