"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add tracking risk constraints to `model`.

The `L1Norm` overload uses an L1-norm cone. The `L2Norm` / `SquaredL2Norm`
overload uses an SOC. The `LpNorm` overload uses power cones parameterised by `r.alg.p`.
The `LInfNorm` overload uses an infinity-norm cone. The independent-variable overload
shifts the weight vector by a benchmark before it builds the inner risk under a nested prefix.
The dependent-variable overload computes a benchmark risk and adds an L1-norm cone on the
risk difference to the inner risk, which it builds under a nested prefix.

# Mathematical definition

```math
\\begin{align}
\\mathrm{TR}_p(\\boldsymbol{w}) &= \\frac{\\lVert \\mathbf{X}\\boldsymbol{w} - \\boldsymbol{b}\\,k \\rVert_p}{c_p}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TR}_p(\\boldsymbol{w})``: Tracking risk measure.
  - ``\\mathbf{X}``: Return matrix ``T \\times N``.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{b}``: Benchmark return series.
  - ``k``: Rebalancing factor (0 or 1).
  - ``c_p``: Normalisation constant depending on the norm order ``p``.

where ``\\boldsymbol{b}`` is the benchmark return series, ``k`` is the budget scaling variable, and ``c_p`` is the norm-order scaling factor, [`norm_factor`](@ref) of `r.alg` (``T - d``, ``\\sqrt{T-d}``, ``1``, etc.).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r`: Tracking risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])

# Returns

  - `nothing`.

# Related

  - [`set_tracking_risk!`](@ref)
  - [`nested_prefix`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:L1Norm},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    T = length(net_X)
    t_tracking_risk = state_set!(model, prefix, :t_tracking_risk_, i, JuMP.@variable(model))
    tracking_risk = state_set!(model, prefix, :tracking_risk_, i,
                               JuMP.@expression(model,
                                                t_tracking_risk / norm_factor(r.alg, T)))
    tr = r.tr
    benchmark = tracking_benchmark(tr, X)
    tracking_r = state_set!(model, prefix, :tracking_r_, i,
                            JuMP.@expression(model,
                                             net_X -
                                             tracking_benchmark_expression(model, benchmark,
                                                                           prefix)))
    state_set!(model, prefix, :ctracking_r_noc_, i,
               JuMP.@constraint(model,
                                [sc * t_tracking_risk;
                                 sc * tracking_r] in JuMP.MOI.NormOneCone(1 + T)))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, :tracking_risk_,
                                    i; prefix = prefix)
    return tracking_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finalise the L2 or squared-L2 tracking risk expression and apply bounds.

The `L2Norm` overload calls [`set_risk_bounds_and_expression!`](@ref) directly with the
SOC variable. The `SquaredL2Norm` overload squares it and applies a sqrt-converted upper
bound to the original SOC variable.

# Arguments

  - $(arg_dict[:model])
  - `r::TrackingRiskMeasure`: Tracking risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - `tracking_risk::JuMP.AbstractJuMPScalar`: Normalised tracking-risk SOC variable.
  - $(arg_dict[:ci])

# Returns

  - `nothing`.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
function set_tracking_risk!(model::JuMP.Model,
                            r::TrackingRiskMeasure{<:Any, <:Any, <:L2Norm},
                            opt::RiskConstraintOwner,
                            tracking_risk::JuMP.AbstractJuMPScalar, i;
                            prefix::Symbol = Symbol(""))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, :tracking_risk_,
                                    i; prefix = prefix)
    return tracking_risk
end
function set_tracking_risk!(model::JuMP.Model,
                            r::TrackingRiskMeasure{<:Any, <:Any, <:SquaredL2Norm},
                            opt::RiskConstraintOwner,
                            tracking_risk::JuMP.AbstractJuMPScalar, i;
                            prefix::Symbol = Symbol(""))
    qtracking_risk = state_set!(model, prefix, :sq_tracking_risk_, i,
                                JuMP.@expression(model, tracking_risk^2))
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, tracking_risk, ub,
                          state_key(prefix, :tracking_risk_, i))
    set_risk_expression!(model, qtracking_risk, r.settings.scale, r.settings.rke)
    return qtracking_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `TrackingRiskMeasure` with `L2Norm` or
`SquaredL2Norm` to `model`.

Introduces a scalar variable and an SOC constraint to encode the L2 (root mean squared)
tracking error between portfolio and benchmark returns. The cone variable is the L2 norm for
both norms, so it is divided by [`norm_factor`](@ref) of an [`L2Norm`](@ref) with the same
`ddof`. [`set_tracking_risk!`](@ref) then squares it for a [`SquaredL2Norm`](@ref).

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::TrackingRiskMeasure{<:Any, <:Any, <:Union{L2Norm, SquaredL2Norm}}`:
    The tracking risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`L2Norm`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any,
                                                      <:Union{L2Norm, SquaredL2Norm}},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    T = length(net_X)
    t_tracking_risk = state_set!(model, prefix, :t_tracking_risk_, i, JuMP.@variable(model))
    tracking_risk = state_set!(model, prefix, :tracking_risk_, i,
                               JuMP.@expression(model,
                                                t_tracking_risk /
                                                norm_factor(L2Norm(; ddof = r.alg.ddof), T)))
    tr = r.tr
    benchmark = tracking_benchmark(tr, X)
    tracking_r = state_set!(model, prefix, :tracking_r_, i,
                            JuMP.@expression(model,
                                             net_X -
                                             tracking_benchmark_expression(model, benchmark,
                                                                           prefix)))
    state_set!(model, prefix, :ctracking_r_soc_, i,
               JuMP.@constraint(model,
                                [sc * t_tracking_risk;
                                 sc * tracking_r] in JuMP.SecondOrderCone()))
    return set_tracking_risk!(model, r, opt, tracking_risk, i; prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `TrackingRiskMeasure` with `LpNorm` to `model`.

Introduces a scalar variable and power-cone constraints to encode the Lp-norm tracking
error between portfolio and benchmark returns, scaled by `(T - ddof)^(1/p)`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::TrackingRiskMeasure{<:Any, <:Any, <:LpNorm}`: The tracking risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`LpNorm`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:LpNorm},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    @argcheck(r.alg.p > 1,
              DomainError(r.alg.p,
                          "`LpNorm.p` is $(r.alg.p), and the tracking risk is the `p`-norm of the deviation, which the model states with a power cone of exponent `1 / p`, so `1 < p` must hold. State a value greater than `1`."))
    sc = get_constraint_scale(model)
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    T = length(net_X)
    t_tracking_risk = state_set!(model, prefix, :t_tracking_risk_, i, JuMP.@variable(model))
    r_tr = state_set!(model, prefix, :r_tracking_risk_, i, JuMP.@variable(model, [1:T]))
    p_inv = inv(r.alg.p)
    scale = norm_factor(r.alg, T)
    tracking_risk = state_set!(model, prefix, :tracking_risk_, i,
                               JuMP.@expression(model, t_tracking_risk / scale))
    benchmark = tracking_benchmark(r.tr, X)
    tracking_r = state_set!(model, prefix, :tracking_r_, i,
                            JuMP.@expression(model,
                                             net_X -
                                             tracking_benchmark_expression(model, benchmark,
                                                                           prefix)))
    state_set!(model, prefix, :ctracking_r_pnorm_, i,
               JuMP.@constraint(model, [i = 1:T],
                                [sc * r_tr[i], sc * t_tracking_risk, sc * tracking_r[i]] in
                                JuMP.MOI.PowerCone(p_inv)))
    state_set!(model, prefix, :ctracking_r_pnorm_eq_, i,
               JuMP.@constraint(model, sc * (sum(r_tr) - t_tracking_risk) == 0))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, :tracking_risk_,
                                    i; prefix = prefix)
    return tracking_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `TrackingRiskMeasure` with `LInfNorm` to `model`.

Introduces a scalar variable and an infinity-norm cone constraint to encode the L∞-norm
(maximum) tracking error between portfolio and benchmark returns. Its [`norm_factor`](@ref) is `1`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::TrackingRiskMeasure{<:Any, <:Any, <:LInfNorm}`: The tracking risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`LInfNorm`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:LInfNorm},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    T = length(net_X)
    t_tracking_risk = state_set!(model, prefix, :t_tracking_risk_, i, JuMP.@variable(model))
    scale = norm_factor(r.alg, T)
    tracking_risk = state_set!(model, prefix, :tracking_risk_, i,
                               JuMP.@expression(model, t_tracking_risk / scale))
    benchmark = tracking_benchmark(r.tr, X)
    tracking_r = state_set!(model, prefix, :tracking_r_, i,
                            JuMP.@expression(model,
                                             net_X -
                                             tracking_benchmark_expression(model, benchmark,
                                                                           prefix)))
    state_set!(model, prefix, :ctracking_infnorm_, i,
               JuMP.@constraint(model,
                                [sc * t_tracking_risk;
                                 sc * tracking_r] in JuMP.MOI.NormInfinityCone(1 + T)))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, :tracking_risk_,
                                    i; prefix = prefix)
    return tracking_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `RiskTrackingRiskMeasure` with `IndependentVariableTracking`
to `model`.

Stores the benchmark-shifted weights `w - wb*k` at `Symbol(tprefix, :w)` under the composed
tracking prefix `tprefix = nested_prefix(prefix, :tr_iv_, i)`, builds the inner risk on
those weights with [`set_risk_constraints!`](@ref) at the index `1` under `tprefix`, then
applies risk bounds and expression registration. The prefix namespacing replaces the former
save/restore swap and is re-entrant.

The inner build reads and writes its Model State keys (`:w`, `:net_X`, the scratch of each
measure) under `tprefix`, so they cannot collide with the keys of the enclosing build. Its
prefix alone separates them, so the index `1` seeds only the measure axis, as at the outer
level, because [`state_key`](@ref) resolves both axes. The lifted matrix `:W` and the marks
`:variance_flag` and `:rc_variance` belong to the namespace that owns the weights,
[`weights_prefix`](@ref). For the shifted weights here, that is `tprefix`. The prefixes
compose through [`nested_prefix`](@ref), so a tracking measure inside a tracking measure is
free of collisions.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any, <:IndependentVariableTracking}`:
    The risk-tracking risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])

# Returns

  - The tracking risk JuMP expression.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`IndependentVariableTracking`](@ref)
  - [`nested_prefix`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                          <:IndependentVariableTracking},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult,
                               pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    ri = r.r
    wb = r.tr.w
    w = get_w(model, prefix)
    k = get_k(model)
    tprefix = nested_prefix(prefix, :tr_iv_, i)
    state_set!(model, tprefix, :w, JuMP.@expression(model, w - wb * k))
    tracking_risk = set_risk_constraints!(model, 1, ri, opt, pr, pl, fees, args...;
                                          prefix = tprefix, kwargs...)
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, :tracking_risk_,
                                    i; prefix = prefix)
    return tracking_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `RiskTrackingRiskMeasure` with `DependentVariableTracking`
to `model`.

Computes the benchmark's expected risk value, stores the (unshifted) portfolio weights at
`Symbol(tprefix, :w)` under the composed tracking prefix
`tprefix = nested_prefix(prefix, :tr_dv_, i)`, builds the inner portfolio risk with
[`set_risk_constraints!`](@ref) at the index `1` under `tprefix`, then adds an L1-norm cone constraint on the difference between the portfolio's
risk expression and the benchmark's expected risk scaled by the allocation variable `k`.
The prefix namespacing replaces the former save/restore swap and is re-entrant.

The weights under `tprefix` are the weights under `prefix`, so the method also registers
`w_owner`, the namespace that owns them, [`weights_prefix`](@ref). The inner build then
reads the enclosing build's lifted matrix ``\\mathbf{W}`` and its marks `variance_flag` and
`rc_variance`. A semidefinite phylogeny on the head constrains the inner variance. The inner
variance is no term of the objective, because the tracking constructor clears its `rke`, so
it does not remove the phylogeny's `p · tr(W)` penalty, [`mark_objective_variance!`](@ref).

# JuMP formulation

## Variables

  - `w`, `k`: read from the model.
  - `tracking_risk_i`: created, the tracking risk.

## Expressions

  - `r_dv_i`: ``d = \\rho(\\boldsymbol{w}) - r_b k``.

## Constraints

  - `crtr_noc_i`: ``[s_c t;\\ s_c d] \\in \\mathcal{K}_{1}``, that is ``t \\geq \\lvert d \\rvert``.

## Relaxation

$(val_dict[:relax])

  - **Direction.** `tracking_risk_i` can lie below ``\\lvert R(\\boldsymbol{w}) - r_b k \\rvert`` when the inner model expression ``\\rho(\\boldsymbol{w})`` lies above the exact risk ``R(\\boldsymbol{w})``. Every inner measure that the model states through an epigraph has ``\\rho(\\boldsymbol{w}) \\geq R(\\boldsymbol{w})``, and the solver can raise ``\\rho`` to ``r_b k`` at no cost. For [`ConditionalValueatRisk`](@ref) it raises the auxiliary variables, for [`StandardDeviation`](@ref) and [`LowOrderMoment`](@ref) the cone variable. A [`Variance`](@ref) in the semidefinite form has ``\\rho = \\mathrm{tr}(\\boldsymbol{\\Sigma}\\mathbf{W})``, and the step ``\\mathbf{W} + t\\,\\boldsymbol{e}_i\\boldsymbol{e}_i^\\intercal`` keeps every row. With no price on that rise, the model value is ``\\max(0,\\ R(\\boldsymbol{w}) - r_b k)``: it penalises a portfolio riskier than the benchmark, and it can report zero for one that is less risky.
  - **Quantity.** `tracking_risk_i` and `r_dv_i`.
  - **Condition.** The model value equals the exact value when ``\\rho(\\boldsymbol{w}) = R(\\boldsymbol{w})`` at the solution. When ``R(\\boldsymbol{w}) \\geq r_b k`` it is not below the exact value. The inner build owns its auxiliary variables under `tprefix`, so no other term prices them, and an epigraph measure stays one-sided. The semidefinite [`Variance`](@ref) is the one exception, because its ``\\mathbf{W}`` is the head's. In the objective, ``\\mathbf{W}`` stays tight when the phylogeny's penalty and the variances that the objective minimises on the same weights price the step at least as much as the tracking term gains, ``p + s_v \\Sigma_{ii} \\geq s \\Sigma_{ii}`` for every ``i``. With no minimised variance that is ``p \\geq s \\max_i \\Sigma_{ii}``. As a bound, `settings.ub`, no condition makes it tight, so the bound is the ceiling ``R(\\boldsymbol{w}) \\leq (r_b + u)\\,k`` for every inner measure.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - $(math_dict[:sc_scale])
  - ``t``: The tracking risk, `tracking_risk_i`.
  - ``\\rho(\\boldsymbol{w})``: The inner measure's model expression, built under `tprefix`.
  - ``R(\\boldsymbol{w})``: The inner measure's exact risk.
  - ``r_b``: The benchmark's expected risk, the inner measure at `r.tr.w`.
  - ``\\mathcal{K}_{1}``: The norm-one cone of dimension 2.
  - ``\\boldsymbol{\\Sigma}``: The covariance matrix of the inner variance.
  - ``\\mathbf{W}``: The lifted matrix of the weights, [`set_sdp_constraints!`](@ref).
  - ``p``: The penalty of the [`SemiDefinitePhylogeny`](@ref), [`set_sdp_phylogeny_constraints!`](@ref).
  - ``s``: The scale of the tracking measure, `r.settings.scale`.
  - ``u``: The upper bound of the tracking measure, `r.settings.ub`.
  - ``s_v``: The scale of a variance that the objective minimises on the same weights.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any, <:DependentVariableTracking}`:
    The risk-tracking risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])

# Returns

  - The tracking risk JuMP expression.

# Related

  - [`RiskTrackingRiskMeasure`](@ref)
  - [`DependentVariableTracking`](@ref)
  - [`weights_prefix`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                          <:DependentVariableTracking},
                               opt::RiskConstraintOwner, pr::AbstractPriorResult,
                               pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    # `r` reached here through `set_risk_constraints!`, which resolved it. `deferred_slots`
    # declares `r` as this container's child, so the derived recursion has already run and
    # `ri` holds values. The benchmark scalar below and the model expression further down are
    # therefore built from one resolution, not two.
    ri = r.r
    wb = r.tr.w
    rb = expected_risk(factory(ri, pr, risk_constraint_solver(opt)), wb, pr.X, fees)
    k = get_k(model)
    sc = get_constraint_scale(model)
    tracking_risk = state_set!(model, prefix, :tracking_risk_, i, JuMP.@variable(model))
    tprefix = nested_prefix(prefix, :tr_dv_, i)
    state_set!(model, tprefix, :w, get_w(model, prefix))
    state_set!(model, tprefix, :w_owner, weights_prefix(model, prefix))
    risk_expr = set_risk_constraints!(model, 1, ri, opt, pr, pl, fees, args...;
                                      prefix = tprefix, kwargs...)
    dr = state_set!(model, prefix, :r_dv_, i, JuMP.@expression(model, risk_expr - rb * k))
    state_set!(model, prefix, :crtr_noc_, i,
               JuMP.@constraint(model,
                                [sc * tracking_risk;
                                 sc * dr] in JuMP.MOI.NormOneCone(2)))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, :tracking_risk_,
                                    i; prefix = prefix)
    return tracking_risk
end
