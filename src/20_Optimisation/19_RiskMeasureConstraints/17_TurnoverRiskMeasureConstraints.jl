"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add turnover risk constraints to `model`.

Introduces a scalar variable `turnover_risk` and the L1-norm cone constraint
`[sc * turnover_risk; sc * (w - benchmark * k)] in NormOneCone(1 + N)` where `benchmark`
is the reference weight vector from `r.w`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{Turnover}(\\boldsymbol{w}) &= \\|\\boldsymbol{w} - \\boldsymbol{w}_b k\\|_1\\,.
\\end{align}
```

Where:

  - ``\\mathrm{Turnover}(\\boldsymbol{w})``: Portfolio turnover.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{w}_b``: Benchmark weight vector.
  - ``k``: Rebalancing factor (0 or 1).
  - ``\\|\\cdot\\|_1``: L1 norm.

where ``\\boldsymbol{w}_b`` is the benchmark weight vector and ``k`` is the budget scaling variable.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::TurnoverRiskMeasure`: Turnover risk measure instance carrying the benchmark weights.
  - $(arg_dict[:opt_rjumpe])

# Returns

  - `nothing`.

# Related

  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::TurnoverRiskMeasure,
                               opt::RiskJuMPOptimisationEstimator, ::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:turnover_risk_, i)
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    k = get_k(model)
    N = length(w)
    turnover_risk = model[key] = JuMP.@variable(model)
    benchmark = r.w
    turnover_r = model[Symbol(:turnover_r_, i)] = JuMP.@expression(model, w - benchmark * k)
    model[Symbol(:cturnover_r_noc_, i)] = JuMP.@constraint(model,
                                                           [sc * turnover_risk;
                                                            sc * turnover_r] in
                                                           JuMP.MOI.NormOneCone(1 + N))
    set_risk_bounds_and_expression!(model, opt, turnover_risk, r.settings, key)
    return turnover_risk
end
