"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add range risk constraints to `model`.

Calls [`set_wr_risk_expression!`](@ref) to obtain the worst-realisation variable, then
introduces a best-realisation variable `br_risk` with constraint
`sc * (br_risk .+ net_X) <= 0`, and defines `range_risk = wr_risk - br_risk`. Returns the
existing expression if already present.

# Mathematical definition

```math
\\begin{align}
\\mathrm{Range}(\\boldsymbol{w}) &= \\max_t(-\\hat{r}_t) - \\min_t(-\\hat{r}_t) = \\mathrm{WR} - \\mathrm{BR}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{Range}(\\boldsymbol{w})``: Return range.
  - ``\\mathrm{WR} = -\\min_t \\hat{r}_t``: Worst realisation.
  - ``\\mathrm{BR} = \\max_t \\hat{r}_t``: Best realisation.
  - ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}``: Portfolio return at time ``t``.

where ``\\mathrm{WR} = -\\min_t \\hat{r}_t`` and ``\\mathrm{BR} = -\\max_t \\hat{r}_t``.

# Arguments

  - $(arg_dict[:model])
  - `r::Range`: Range risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`set_wr_risk_expression!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::Range,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    if haskey(model, :range_risk)
        return model[:range_risk]
    end
    sc = model[:sc]
    wr_risk = set_wr_risk_expression!(model, pr.X)
    net_X = model[:net_X]
    JuMP.@variable(model, br_risk)
    JuMP.@expression(model, range_risk, wr_risk - br_risk)
    JuMP.@constraint(model, cbr, sc * (br_risk .+ net_X) <= 0)
    set_risk_bounds_and_expression!(model, opt, range_risk, r.settings, :range_risk)
    return range_risk
end
