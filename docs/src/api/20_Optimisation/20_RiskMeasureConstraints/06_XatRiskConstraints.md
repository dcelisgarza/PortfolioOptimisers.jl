# XatRisk Constraints

```@docs
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRisk{<:Any, <:Any, <:Any, <:MIPValueatRisk}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:MIPValueatRisk}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:DistributionValueatRisk}, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::DrawdownatRisk, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
set_mip_quantile_risk_constraints!
```
