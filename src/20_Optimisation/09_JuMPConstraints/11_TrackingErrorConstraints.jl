"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add tracking error constraints to the JuMP optimisation model.

The fall-through method does nothing. Concrete methods dispatch on the tracking algorithm type:

  - [`L1Norm`](@ref): Enforces `‖net_X - wb * k‖₁ ≤ err * T` via NormOneCone.
  - [`L2Norm`](@ref): Enforces a scaled L2 norm via SecondOrderCone.
  - [`SquaredL2Norm`](@ref): The same cone, with the bound square-rooted, because `err` bounds the *squared* error that [`norm_error`](@ref) reports.
  - [`LpNorm`](@ref): Enforces a scaled Lp norm via power cone.
  - [`LInfNorm`](@ref): Enforces `‖net_X - wb * k‖_∞ ≤ err * scale` via NormInfinityCone.
  - [`IndependentVariableTracking`](@ref): Substitutes `w - wb` for `w` and applies the chosen risk constraint.
  - [`DependentVariableTracking`](@ref): Constrains the absolute difference between portfolio risk and benchmark risk.

The collection method iterates over all tracking errors in `tres`.

# Mathematical definition

```math
\\begin{align}
t_{te} &\\geq \\lVert \\mathbf{X}\\boldsymbol{w} - \\boldsymbol{b} k \\rVert_p \\cdot c_p^{-1}\\,, \\\\
t_{te} &\\leq \\mathrm{err} \\cdot k\\,.
\\end{align}
```

Where:

  - ``t_{te}``: Auxiliary tracking error scalar variable.
  - ``\\mathbf{X}``: Asset returns matrix (``T \\times N``).
  - $(math_dict[:w_port])
  - ``\\boldsymbol{b}``: Benchmark return vector.
  - $(math_dict[:k_budget])
  - $(math_dict[:p_norm_order])
  - ``c_p``: Norm-specific scaling factor (``T``, ``\\sqrt{T - d}``, etc.).
  - ``\\mathrm{err}``: Tracking error tolerance.

# Arguments

  - $(arg_dict[:model])
  - `i::Integer`: Constraint index for generating unique variable and constraint names.
  - `pr::AbstractPriorResult`: Prior result providing the return matrix `X`.
  - `tr`: Tracking error specification.
  - `opt`: Optimisation estimator (required for risk-based tracking variants).
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])

# Returns

  - `nothing`.

# Related

  - [`TrackingError`](@ref)
  - [`RiskTrackingError`](@ref)
  - [`L1Norm`](@ref)
  - [`L2Norm`](@ref)
  - [`LpNorm`](@ref)
  - [`LInfNorm`](@ref)
"""
function set_tracking_error_constraints!(args...; kwargs...)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::TrackingError{<:Any, <:Any, <:L1Norm}, args...;
                                         kwargs...)
    X = pr.X
    k = get_k(model)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(tr.tr, X)
    err = tr.err
    T = size(X, 1)
    f = err * T
    t_te = state_set!(model, Symbol(""), :t_te_, i, JuMP.@variable(model))
    tr = state_set!(model, Symbol(""), :te_, i, JuMP.@expression(model, net_X - wb * k))
    cte_noc, cte = JuMP.@constraints(model,
                                     begin
                                         [sc * t_te;
                                          sc * tr] in JuMP.MOI.NormOneCone(1 + T)
                                         sc * (t_te - f * k) <= 0
                                     end)
    state_set!(model, Symbol(""), :cte_noc_, i, cte_noc)
    state_set!(model, Symbol(""), :cte_, i, cte)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Convert a [`TrackingError`](@ref) tolerance into the bound on the second-order cone variable.

Both norms share one cone, which bounds ``\\lVert \\mathbf{X}\\boldsymbol{w} - \\boldsymbol{b}k \\rVert_2``. They do not share the quantity that `err` bounds. [`norm_error`](@ref) divides that norm by ``\\sqrt{T - d}`` for an [`L2Norm`](@ref) and squares it before dividing by ``T - d`` for a [`SquaredL2Norm`](@ref), so the second bound is square-rooted here to keep the model, the functor and `set_risk_constraints!` in agreement.

# Arguments

  - `f`: The [`NormError`](@ref) the tracking error carries.
  - `err::Number`: Tracking error tolerance.
  - `T::Integer`: Number of observations.

# Returns

  - `f::Number`: Upper bound on the cone variable, before the budget scaling by ``k``.

# Related

  - [`set_tracking_error_constraints!`](@ref)
  - [`TrackingError`](@ref)
  - [`norm_error`](@ref)
  - [`L2Norm`](@ref)
  - [`SquaredL2Norm`](@ref)
"""
function tracking_error_soc_factor(f::L2Norm, err::Number, T::Integer)
    return err * sqrt(T - f.ddof)
end
function tracking_error_soc_factor(f::SquaredL2Norm, err::Number, T::Integer)
    return sqrt(err * (T - f.ddof))
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::TrackingError{<:Any, <:Any,
                                                           <:Union{<:L2Norm,
                                                                   <:SquaredL2Norm}},
                                         args...; kwargs...)
    X = pr.X
    k = get_k(model)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(tr.tr, X)
    err = tr.err
    f = tracking_error_soc_factor(tr.alg, err, size(X, 1))
    t_te = state_set!(model, Symbol(""), :t_te_, i, JuMP.@variable(model))
    tr = state_set!(model, Symbol(""), :te_, i, JuMP.@expression(model, net_X - wb * k))
    cte_soc, cte = JuMP.@constraints(model,
                                     begin
                                         [sc * t_te;
                                          sc * tr] in JuMP.SecondOrderCone()
                                         sc * (t_te - f * k) <= 0
                                     end)
    state_set!(model, Symbol(""), :cte_soc_, i, cte_soc)
    state_set!(model, Symbol(""), :cte_, i, cte)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::TrackingError{<:Any, <:Any, <:LpNorm}, args...;
                                         kwargs...)
    @argcheck(tr.alg.p > 1, DomainError)
    X = pr.X
    k = get_k(model)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(tr.tr, X)
    T = size(X, 1)
    err = tr.err
    p_inv = inv(tr.alg.p)
    scale = T - tr.alg.ddof
    f = err * (tr.alg.p == 3 ? cbrt(scale) : scale^p_inv)
    t_te, r_te = JuMP.@variables(model, begin
                                     ()
                                     [1:T]
                                 end)
    state_set!(model, Symbol(""), :t_te_, i, t_te)
    state_set!(model, Symbol(""), :r_te_, i, r_te)
    tr = state_set!(model, Symbol(""), :te_, i, JuMP.@expression(model, net_X - wb * k))
    cte_pnorm, cste, cte = JuMP.@constraints(model,
                                             begin
                                                 [i = 1:T],
                                                 [sc * r_te[i], sc * t_te, sc * tr[i]] in
                                                 JuMP.MOI.PowerCone(p_inv)
                                                 sc * (sum(r_te) - t_te) == 0
                                                 sc * (t_te - f * k) <= 0
                                             end)
    state_set!(model, Symbol(""), :cte_pnorm_, i, cte_pnorm)
    state_set!(model, Symbol(""), :cste_, i, cste)
    state_set!(model, Symbol(""), :cte_, i, cte)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::TrackingError{<:Any, <:Any, <:LInfNorm},
                                         args...; kwargs...)
    X = pr.X
    k = get_k(model)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X)
    wb = tracking_benchmark(tr.tr, X)
    T = size(X, 1)
    err = tr.err
    scale = T - tr.alg.ddof
    f = err * scale
    t_te = state_set!(model, Symbol(""), :t_te_, i, JuMP.@variable(model))
    tr = state_set!(model, Symbol(""), :te_, i, JuMP.@expression(model, net_X - wb * k))
    cte_infnorm, cte = JuMP.@constraints(model,
                                         begin
                                             [sc * t_te
                                              sc * tr] in JuMP.MOI.NormInfinityCone(1 + T)
                                             sc * (t_te - f * k) <= 0
                                         end)
    state_set!(model, Symbol(""), :cte_infnorm_, i, cte_infnorm)
    state_set!(model, Symbol(""), :cte_, i, cte)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::RiskTrackingError{<:Any, <:Any, <:Any,
                                                               <:IndependentVariableTracking},
                                         opt::JuMPOptimisationEstimator,
                                         pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees},
                                         args...; prefix::Symbol = Symbol(""), kwargs...)
    r = tr.r
    wb = tr.tr.w
    err = tr.err
    w = get_w(model, prefix)
    k = get_k(model)
    sc = get_constraint_scale(model)
    tprefix = nested_prefix(prefix, :te_ir_, i)
    state_set!(model, tprefix, :w, JuMP.@expression(model, w - wb * k))
    risk_expr = set_risk_tracking_risk_constraints!(model, r, opt, pr, pl, fees, tprefix,
                                                    args...; kwargs...)
    state_set!(model, prefix, :cter_, i,
               JuMP.@constraint(model, sc * (risk_expr - err * k) <= 0))
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::RiskTrackingError{<:Any, <:Any, <:Any,
                                                               <:DependentVariableTracking},
                                         opt::JuMPOptimisationEstimator,
                                         pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees},
                                         args...; prefix::Symbol = Symbol(""), kwargs...)
    ri = tr.r
    wb = tr.tr.w
    err = tr.err
    rb = expected_risk(factory(ri, pr, opt.opt.slv), wb, pr.X, fees)
    k = get_k(model)
    sc = get_constraint_scale(model)
    te_dr = state_set!(model, prefix, :te_dr_, i, JuMP.@variable(model))
    tprefix = nested_prefix(prefix, :te_dr_, i)
    state_set!(model, tprefix, :w, get_w(model, prefix))
    risk_expr = set_risk_tracking_risk_constraints!(model, ri, opt, pr, pl, fees, tprefix,
                                                    args...; kwargs...)
    # The risk difference is its own entry name. It was `Symbol(key, i)` — the *composed*
    # key with the index appended a second time — which put entry 1 at `:te_dr_11`, the
    # key `te_dr` itself takes at entry 11.
    dr = state_set!(model, prefix, :te_dr_diff_, i,
                    JuMP.@expression(model, risk_expr - rb * k))
    cter_noc, cter = JuMP.@constraints(model,
                                       begin
                                           [sc * te_dr
                                            sc * dr] in JuMP.MOI.NormOneCone(2)
                                           sc * (te_dr - err * k) <= 0
                                       end)
    state_set!(model, prefix, :cter_noc_, i, cter_noc)
    state_set!(model, prefix, :cter_, i, cter)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, pr::AbstractPriorResult,
                                         tres::Tr_VecTr, args...; kwargs...)
    for (i, tr) in enumerate(tres)
        set_tracking_error_constraints!(model, i, pr, tr, args...; kwargs...)
    end
    return nothing
end
