```@meta
Description = "Kurtosis Constraints, private API of PortfolioOptimisers.jl: get_chol_or_Gkt_pm, get_kt_Akt_pm, set_kurtosis_risk!, set_risk_constraints!."
```

# Kurtosis Constraints: private API

```@docs
get_chol_or_Gkt_pm
get_kt_Akt_pm
set_kurtosis_risk!
set_risk_constraints!(model::JuMP.Model, i::Any, r::Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Integer, <:Any, <:Any}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
set_risk_constraints!(model::JuMP.Model, i::Any, r::Kurtosis{<:Any, <:Any, <:Any, <:Any, Nothing, <:Any, <:Any}, opt::RiskConstraintOwner, pr::AbstractPriorResult, args...; kwargs...)
```
