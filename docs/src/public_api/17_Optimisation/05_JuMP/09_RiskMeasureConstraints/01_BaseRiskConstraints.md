```@meta
Description = "Base Risk Constraints, public API of PortfolioOptimisers.jl: scalarise_risk_expression!, risk_constraint_solver, set_risk_constraints!, …"
```

# Base Risk Constraints

```@docs
scalarise_risk_expression!
risk_constraint_solver(opt::JuMPOptimisationEstimator)
set_risk_constraints!(model::JuMP.Model, r::RiskMeasure, opt::Union{JuMPOptimisationEstimator, AbstractProgrammeAllocationSet}, pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, b1::Option{<:MatNum} = nothing; kwargs...)
set_risk_bounds_and_expression!
```
