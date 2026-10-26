"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the epigraph of the turnover of a [`TurnoverRiskMeasure`](@ref) to the model, and return its risk variable.

One norm-one cone bounds the risk variable from below by the ``L^1`` distance of the weights from the reference weights `r.w`, scaled by ``k``. The risk expression of the measure is that variable. Outside a ratio problem ``k = 1``. In a ratio problem the weights are homogenised, and the row bounds ``k`` times the turnover of ``\\boldsymbol{w} / k``. The model reads no moment, so it accepts any prior result and does not read it.

# JuMP formulation

## Variables

  - `w`: read, ``\\boldsymbol{w}``.
  - `k`: read, ``k``.
  - `turnover_risk_<i>`: created, ``t``.

## Expressions

  - `turnover_r_<i>`: ``\\boldsymbol{w} - \\boldsymbol{w}_{0} k``.

## Constraints

  - `cturnover_r_noc_<i>`: ``(s_c t,\\, s_c (\\boldsymbol{w} - \\boldsymbol{w}_{0} k)) \\in \\mathcal{K}_{1}``, which is ``t \\geq \\lVert \\boldsymbol{w} - \\boldsymbol{w}_{0} k \\rVert_1``.
  - `turnover_risk_<i>_ub`: ``s_c (t - u k) \\leq 0``, when `r.settings.ub` is a number.

[`set_risk_bounds_and_expression!`](@ref) registers the bound row and adds ``t`` to `risk_vec` when `r.settings.rke` is `true`. Each name carries the index `i` of the measure and the namespace `prefix`.

Where:

  - ``t``: Epigraph variable of the turnover, the risk of the measure.
  - $(math_dict[:w_port])
  - $(math_dict[:w_0_turnover])
  - $(math_dict[:k_budget])
  - $(math_dict[:sc_scale])
  - $(math_dict[:K_q_norm])
  - ``u``: `r.settings.ub`.

## Relaxation

$(val_dict[:relax])

  - The cone bounds ``t`` from below only, so ``t`` is the turnover only when the row is tight.
  - When the objective minimises the risk, the row is tight. A risk that the objective does not minimise can lie above the turnover, but the bound row then still holds the turnover, because the turnover is at or below ``t``.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::TurnoverRiskMeasure`: Turnover risk measure. Its `w` holds the reference weights.
  - $(arg_dict[:opt_rjumpe])
  - `::AbstractPriorResult`: Prior result. The model does not read it.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `turnover_risk`: The epigraph variable ``t``.

# Related

  - [`TurnoverRiskMeasure`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
  - [`set_risk_upper_bound!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::TurnoverRiskMeasure,
                               opt::RiskConstraintOwner, ::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    k = get_k(model)
    N = length(w)
    turnover_risk = state_set!(model, prefix, :turnover_risk_, i, JuMP.@variable(model))
    benchmark = r.w
    turnover_r = state_set!(model, prefix, :turnover_r_, i,
                            JuMP.@expression(model, w - benchmark * k))
    state_set!(model, prefix, :cturnover_r_noc_, i,
               JuMP.@constraint(model,
                                [sc * turnover_risk; sc * turnover_r] in
                                JuMP.MOI.NormOneCone(1 + N)))
    set_risk_bounds_and_expression!(model, opt, turnover_risk, r.settings, :turnover_risk_,
                                    i; prefix = prefix)
    return turnover_risk
end
