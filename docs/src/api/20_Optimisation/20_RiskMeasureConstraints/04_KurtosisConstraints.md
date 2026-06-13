# Kurtosis Constraints

```@docs
get_chol_or_Gkt_pm
get_kt_Akt_pm
set_kurtosis_risk!
set_risk_constraints!(model::JuMP.Model, i::Any, r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Integer, <:Any, <:Any}, opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::Kurtosis{<:Any, <:Any, <:Any, <:Any, Nothing, <:Any, <:Any}, opt::RiskJuMPOptimisationEstimator, pr::HighOrderPrior, args...; kwargs...)
set_risk_constraints!(::JuMP.Model, ::Any, ::Kurtosis, ::RiskJuMPOptimisationEstimator, pr::LowOrderPrior, args...; kwargs...)
```
