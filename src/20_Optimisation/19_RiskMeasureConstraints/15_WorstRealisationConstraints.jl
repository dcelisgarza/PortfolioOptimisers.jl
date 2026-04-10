"""
    set_wr_risk_expression!(model, X)

Introduce the worst-realisation risk variable and constraint to `model`.

Creates a scalar variable `wr_risk` and adds `sc * (wr_risk .+ net_X) >= 0` so that
`wr_risk` upper-bounds the negative of every portfolio return. Returns the existing variable
if already present.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix (`T × N`).

# Related

  - [`set_risk_constraints!`](@ref)
"""
function set_wr_risk_expression!(model::JuMP.Model, X::MatNum)
    if haskey(model, :wr_risk)
        return model[:wr_risk]
    end
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    JuMP.@variable(model, wr_risk)
    JuMP.@constraint(model, cwr, sc * (wr_risk .+ net_X) >= 0)
    return wr_risk
end
"""
    set_risk_constraints!(model, ::Any, r::WorstRealisation, opt, pr, args...; kwargs...)

Add worst-realisation risk constraints to `model`.

Delegates to [`set_wr_risk_expression!`](@ref) to create `wr_risk`, then calls
[`set_risk_bounds_and_expression!`](@ref). Returns the existing expression if already present.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `r::WorstRealisation`: Worst-realisation risk measure instance.
  - `opt::RiskJuMPOptimisationEstimator`: Optimisation estimator.
  - `pr::AbstractPriorResult`: Prior result containing `X`.

# Related

  - [`set_wr_risk_expression!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::WorstRealisation,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :wr_risk)
        return model[:wr_risk]
    end
    wr_risk = set_wr_risk_expression!(model, pr.X)
    set_risk_bounds_and_expression!(model, opt, wr_risk, r.settings, :wr_risk)
    return wr_risk
end
