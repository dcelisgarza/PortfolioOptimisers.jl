"""
$(DocStringExtensions.TYPEDSIGNATURES)

Introduce the worst-realisation risk variable and constraint to `model`.

Creates a scalar variable `wr_risk` and adds `sc * (wr_risk .+ net_X) >= 0` so that
`wr_risk` upper-bounds the negative of every portfolio return. Returns the existing variable
if already present.

# Mathematical definition

```math
\\begin{align}
\\mathrm{WR}(\\boldsymbol{w}) &= \\max_{t} (-\\hat{r}_t) = -\\min_{t} \\hat{r}_t\\,.
\\end{align}
```

Where:

  - ``\\mathrm{WR}(\\boldsymbol{w})``: Worst realisation.
  - ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}``: Portfolio return at time ``t``.

where ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}`` is the net portfolio return at time ``t``.

# Arguments

  - $(arg_dict[:model])
  - `X::MatNum`: Asset returns matrix (`T × N`).

# Returns

  - `wr_risk`: JuMP scalar variable for the worst-realisation risk.

# Related

  - [`set_risk_constraints!`](@ref)
"""
function set_wr_risk_expression!(model::JuMP.Model, X::MatNum; prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :wr_risk))
        return model[Symbol(prefix, :wr_risk)]
    end
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    wr_risk = preg!(model, prefix, :wr_risk, JuMP.@variable(model))
    preg!(model, prefix, :cwr, JuMP.@constraint(model, sc * (wr_risk .+ net_X) >= 0))
    return wr_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add worst-realisation risk constraints to `model`.

Delegates to [`set_wr_risk_expression!`](@ref) to create `wr_risk`, then calls
[`set_risk_bounds_and_expression!`](@ref). Returns the existing expression if already present.

# Arguments

  - $(arg_dict[:model])
  - `r::WorstRealisation`: Worst-realisation risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`set_wr_risk_expression!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::WorstRealisation,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    if haskey(model, Symbol(prefix, :wr_risk))
        return model[Symbol(prefix, :wr_risk)]
    end
    wr_risk = set_wr_risk_expression!(model, pr.X; prefix = prefix)
    set_risk_bounds_and_expression!(model, opt, wr_risk, r.settings,
                                    Symbol(prefix, :wr_risk))
    return wr_risk
end
