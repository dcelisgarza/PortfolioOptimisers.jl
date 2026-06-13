"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add tracking risk constraints to `model`.

The `L1Tracking` overload uses an L1-norm cone. The `L2Tracking` / `SquaredL2Tracking`
overload uses an SOC. The `LpTracking` overload uses power cones parameterised by `r.alg.p`.
The `LInfTracking` overload uses an infinity-norm cone. The independent-variable overload
shifts the weight vector by a benchmark before delegating to [`set_risk_tracking_risk_constraints!`](@ref).
The dependent-variable overload computes a benchmark risk and adds an L1-norm cone on the
risk difference via [`set_risk_tracking_risk_constraints!`](@ref).

# Mathematical definition

```math
\\begin{align}
\\mathrm{TR}_p(\\boldsymbol{w}) &= \\frac{\\|\\mathbf{X}\\boldsymbol{w} - \\boldsymbol{b}\\,k\\|_p}{c_p}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{TR}_p(\\boldsymbol{w})``: Tracking risk measure.
  - ``\\mathbf{X}``: Return matrix ``T \\times N``.
  - $(math_dict[:w_port])
  - ``\\boldsymbol{b}``: Benchmark return series.
  - ``k``: Rebalancing factor (0 or 1).
  - ``c_p``: Normalisation constant depending on the norm order ``p``.

where ``\\boldsymbol{b}`` is the benchmark return series, ``k`` is the budget scaling variable, and ``c_p`` is the norm-order scaling factor (``T``, ``\\sqrt{T-d}``, etc.).

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
  - [`set_risk_tr_constraints!`](@ref)
  - [`set_risk_tracking_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:L1Tracking},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:tracking_risk_, i)
    sc = get_constraint_scale(model)
    k = get_k(model)
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    T = length(net_X)
    t_tracking_risk = model[Symbol(:t_tracking_risk_, i)] = JuMP.@variable(model)
    tracking_risk = model[key] = JuMP.@expression(model, t_tracking_risk / T)
    tr = r.tr
    benchmark = tracking_benchmark(tr, X)
    tracking_r = model[Symbol(:tracking_r_, i)] = JuMP.@expression(model,
                                                                   net_X - benchmark * k)
    model[Symbol(:ctracking_r_noc_, i)] = JuMP.@constraint(model,
                                                           [sc * t_tracking_risk;
                                                            sc * tracking_r] in
                                                           JuMP.MOI.NormOneCone(1 + T))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Finalise the L2 or squared-L2 tracking risk expression and apply bounds.

The `L2Tracking` overload calls [`set_risk_bounds_and_expression!`](@ref) directly with the
SOC variable. The `SquaredL2Tracking` overload squares it and applies a sqrt-converted upper
bound to the original SOC variable.

# Arguments

  - $(arg_dict[:model])
  - `r::TrackingRiskMeasure`: Tracking risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - `tracking_risk::JuMP.AbstractJuMPScalar`: Normalised tracking-risk SOC variable.
  - $(arg_dict[:key_sym])

# Returns

  - `nothing`.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
function set_tracking_risk!(model::JuMP.Model,
                            r::TrackingRiskMeasure{<:Any, <:Any, <:L2Tracking},
                            opt::RiskJuMPOptimisationEstimator,
                            tracking_risk::JuMP.AbstractJuMPScalar, key::Symbol)
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
function set_tracking_risk!(model::JuMP.Model,
                            r::TrackingRiskMeasure{<:Any, <:Any, <:SquaredL2Tracking},
                            opt::RiskJuMPOptimisationEstimator,
                            tracking_risk::JuMP.AbstractJuMPScalar, key::Symbol)
    qtracking_risk = model[Symbol(:sq_, key)] = JuMP.@expression(model, tracking_risk^2)
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    set_risk_upper_bound!(model, opt, tracking_risk, ub, key)
    set_risk_expression!(model, qtracking_risk, r.settings.scale, r.settings.rke)
    return qtracking_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `TrackingRiskMeasure` with `L2Tracking` or
`SquaredL2Tracking` to `model`.

Introduces a scalar variable and an SOC constraint to encode the L2 (root mean squared)
tracking error between portfolio and benchmark returns.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::TrackingRiskMeasure{<:Any, <:Any, <:Union{<:L2Tracking, <:SquaredL2Tracking}}`:
    The tracking risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`L2Tracking`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any,
                                                      <:Union{<:L2Tracking,
                                                              <:SquaredL2Tracking}},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:tracking_risk_, i)
    sc = get_constraint_scale(model)
    k = get_k(model)
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    T = length(net_X)
    t_tracking_risk = model[Symbol(:t_tracking_risk_, i)] = JuMP.@variable(model)
    tracking_risk = model[key] = JuMP.@expression(model,
                                                  t_tracking_risk / sqrt(T - r.alg.ddof))
    tr = r.tr
    benchmark = tracking_benchmark(tr, X)
    tracking_r = model[Symbol(:tracking_r_, i)] = JuMP.@expression(model,
                                                                   net_X - benchmark * k)
    model[Symbol(:ctracking_r_soc_, i)] = JuMP.@constraint(model,
                                                           [sc * t_tracking_risk;
                                                            sc * tracking_r] in
                                                           JuMP.SecondOrderCone())
    return set_tracking_risk!(model, r, opt, tracking_risk, key)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `TrackingRiskMeasure` with `LpTracking` to `model`.

Introduces a scalar variable and power-cone constraints to encode the Lp-norm tracking
error between portfolio and benchmark returns, scaled by `(T - ddof)^(1/p)`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::TrackingRiskMeasure{<:Any, <:Any, <:LpTracking}`: The tracking risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`LpTracking`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:LpTracking},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    @argcheck(r.alg.p > 1, DomainError)
    key = Symbol(:tracking_risk_, i)
    sc = get_constraint_scale(model)
    k = get_k(model)
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    T = length(net_X)
    t_tracking_risk, r_tr = model[Symbol(:t_tracking_risk_, i)], model[Symbol(:r_tracking_risk_, i)] = JuMP.@variables(model,
                                                                                                                       begin
                                                                                                                           ()
                                                                                                                           [1:T]
                                                                                                                       end)
    p_inv = inv(r.alg.p)
    scale = T - r.alg.ddof
    scale = r.alg.p == 3 ? cbrt(scale) : scale^p_inv
    tracking_risk = model[key] = JuMP.@expression(model, t_tracking_risk / scale)
    benchmark = tracking_benchmark(r.tr, X)
    tracking_r = model[Symbol(:tracking_r_, i)] = JuMP.@expression(model,
                                                                   net_X - benchmark * k)
    model[Symbol(:ctracking_r_pnorm_, i)], model[Symbol(:ctracking_r_pnorm_eq_, i)] = JuMP.@constraints(model,
                                                                                                        begin
                                                                                                            [i = 1:T],
                                                                                                            [sc *
                                                                                                             r_tr[i],
                                                                                                             sc *
                                                                                                             t_tracking_risk,
                                                                                                             sc *
                                                                                                             tracking_r[i]] in
                                                                                                            JuMP.MOI.PowerCone(p_inv)
                                                                                                            sc *
                                                                                                            (sum(r_tr) -
                                                                                                             t_tracking_risk) ==
                                                                                                            0
                                                                                                        end)
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `TrackingRiskMeasure` with `LInfTracking` to `model`.

Introduces a scalar variable and an infinity-norm cone constraint to encode the L∞-norm
(maximum) tracking error between portfolio and benchmark returns, scaled by `T - ddof`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::TrackingRiskMeasure{<:Any, <:Any, <:LInfTracking}`: The tracking risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`TrackingRiskMeasure`](@ref)
  - [`LInfTracking`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::TrackingRiskMeasure{<:Any, <:Any, <:LInfTracking},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:tracking_risk_, i)
    sc = get_constraint_scale(model)
    k = get_k(model)
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    T = length(net_X)
    t_tracking_risk = model[Symbol(:t_tracking_risk_, i)] = JuMP.@variable(model)
    scale = T - r.alg.ddof
    tracking_risk = model[key] = JuMP.@expression(model, t_tracking_risk / scale)
    benchmark = tracking_benchmark(r.tr, X)
    tracking_r = model[Symbol(:tracking_r_, i)] = JuMP.@expression(model,
                                                                   net_X - benchmark * k)
    model[Symbol(:ctracking_infnorm_, i)] = JuMP.@constraint(model,
                                                             [sc * t_tracking_risk;
                                                              sc * tracking_r] in
                                                             JuMP.MOI.NormInfinityCone(1 +
                                                                                       T))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Dispatch to indexed [`set_risk_constraints!`](@ref) for a single measure or iterate over a
vector of measures, using a name prefix `key` for unique constraint naming.

# Arguments

  - `key`: Name prefix for unique constraint symbols.
  - $(arg_dict[:model])
  - `r`: A [`RiskMeasure`](@ref) or a vector of risk measures.
  - $(arg_dict[:opt_jumpe])
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])

# Returns

  - `nothing`.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_risk_tracking_risk_constraints!`](@ref)
"""
function set_risk_tr_constraints!(key::Any, model::JuMP.Model, r::RiskMeasure,
                                  opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                                  pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                                  kwargs...)
    return set_risk_constraints!(model, Symbol(key, 1), r, opt, pr, pl, fees, args...;
                                 kwargs...)
end
function set_risk_tr_constraints!(key::Any, model::JuMP.Model, rs::VecRM,
                                  opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                                  pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                                  kwargs...)
    for (i, r) in enumerate(rs)
        set_risk_constraints!(model, Symbol(key, i), r, opt, pr, pl, fees, args...;
                              kwargs...)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the inner risk expression for risk tracking under a namespaced `tprefix`.

The caller stores the tracking-difference weights at `Symbol(tprefix, :w)`, so the inner
[`set_risk_tr_constraints!`](@ref) build reads and writes ALL of its shared model-state
keys (`:w`, `:net_X`, `:W`, `:variance_flag`, …) under `tprefix` and cannot collide with
the outer model's keys. This replaces the former save/unregister/restore swap (ADR 0005):
the prefix isolates the nested build structurally. Because tracking prefixes COMPOSE
(`tprefix = Symbol(prefix, :tr_iv_, i, :_)`), tracking-nested-in-tracking is collision-free.

# Arguments

  - $(arg_dict[:model])
  - `r`: Inner risk measure (or vector of measures).
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])
  - `tprefix::Symbol`: Composed tracking prefix namespacing the nested build's keys.

# Returns

  - The inner risk JuMP expression.

# Related

  - [`set_risk_tr_constraints!`](@ref)
"""
function set_risk_tracking_risk_constraints!(model::JuMP.Model, r,
                                             opt::RiskJuMPOptimisationEstimator,
                                             pr::AbstractPriorResult,
                                             pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees},
                                             tprefix::Symbol, args...; kwargs...)
    return set_risk_tr_constraints!(tprefix, model, r, opt, pr, pl, fees, args...;
                                    prefix = tprefix, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `RiskTrackingRiskMeasure` with `IndependentVariableTracking`
to `model`.

Stores the benchmark-shifted weights `w - wb*k` at `Symbol(tprefix, :w)` under the composed
tracking prefix `tprefix = Symbol(prefix, :tr_iv_, i, :_)`, delegates to
[`set_risk_tracking_risk_constraints!`](@ref) to build the inner risk on those weights under
`tprefix`, then applies risk bounds and expression registration. The prefix namespacing
replaces the former save/restore swap (ADR 0005) and is re-entrant.

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
  - [`set_risk_tracking_risk_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                          <:IndependentVariableTracking},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:tracking_risk_, i)
    ri = r.r
    wb = r.tr.w
    w = get_w(model, prefix)
    k = get_k(model)
    tprefix = Symbol(prefix, :tr_iv_, i, :_)
    preg!(model, tprefix, :w, JuMP.@expression(model, w - wb * k))
    tracking_risk = set_risk_tracking_risk_constraints!(model, ri, opt, pr, pl, fees,
                                                        tprefix, args...; kwargs...)
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `RiskTrackingRiskMeasure` with `DependentVariableTracking`
to `model`.

Computes the benchmark's expected risk value, stores the (unshifted) portfolio weights at
`Symbol(tprefix, :w)` under the composed tracking prefix
`tprefix = Symbol(prefix, :tr_dv_, i, :_)`, delegates to
[`set_risk_tracking_risk_constraints!`](@ref) to build the inner portfolio risk under
`tprefix`, then adds an L1-norm cone constraint on the difference between the portfolio's
risk expression and the benchmark's expected risk scaled by the allocation variable `k`.
The prefix namespacing replaces the former save/restore swap (ADR 0005) and is re-entrant.

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
  - [`set_risk_tracking_risk_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::RiskTrackingRiskMeasure{<:Any, <:Any, <:Any,
                                                          <:DependentVariableTracking},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:tracking_risk_, i)
    ri = r.r
    wb = r.tr.w
    rb = expected_risk(factory(ri, pr, opt.opt.slv), wb, pr.X, fees)
    k = get_k(model)
    sc = get_constraint_scale(model)
    tracking_risk = model[key] = JuMP.@variable(model)
    tprefix = Symbol(prefix, :tr_dv_, i, :_)
    preg!(model, tprefix, :w, get_w(model, prefix))
    risk_expr = set_risk_tracking_risk_constraints!(model, ri, opt, pr, pl, fees, tprefix,
                                                    args...; kwargs...)
    dr = model[Symbol(:r_dv_, i)] = JuMP.@expression(model, risk_expr - rb * k)
    model[Symbol(:crtr_noc_, i)] = JuMP.@constraint(model,
                                                    [sc * tracking_risk;
                                                     sc * dr] in JuMP.MOI.NormOneCone(2))
    set_risk_bounds_and_expression!(model, opt, tracking_risk, r.settings, key)
    return tracking_risk
end
