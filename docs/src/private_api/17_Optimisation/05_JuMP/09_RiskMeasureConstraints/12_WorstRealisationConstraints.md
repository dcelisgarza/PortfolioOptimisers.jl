```@meta
Description = "Worst Realisation Constraints, private API of PortfolioOptimisers.jl: set_wr_risk_expression!, set_risk_constraints!."
```

# Worst Realisation Constraints: private API

```@docs
set_wr_risk_expression!
set_risk_constraints!(model::JuMP.Model, ::Any, r::WorstRealisation, opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...; kwargs...)
```
