"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add tracking error constraints to the JuMP optimisation model.

The fall-through method does nothing. Concrete methods dispatch on the tracking algorithm type:

  - [`L1Norm`](@ref): Enforces `‖net_X - wb * k‖₁ ≤ err * T` via NormOneCone.
  - [`L2Norm`](@ref) / [`SquaredL2Norm`](@ref): Enforces a scaled L2 norm via SecondOrderCone.
  - [`LpNorm`](@ref): Enforces a scaled Lp norm via power cone.
  - [`LInfNorm`](@ref): Enforces `‖net_X - wb * k‖_∞ ≤ err * scale` via NormInfinityCone.
  - [`IndependentVariableTracking`](@ref): Substitutes `w - wb` for `w` and applies the chosen risk constraint.
  - [`DependentVariableTracking`](@ref): Constrains the absolute difference between portfolio risk and benchmark risk.

The collection method iterates over all tracking errors in `tres`.

# Mathematical definition

```math
\\begin{align}
t_{te} &\\geq \\|\\mathbf{X}\\boldsymbol{w} - \\boldsymbol{b} k\\|_p \\cdot c_p^{-1}\\,, \\\\
t_{te} &\\leq \\mathrm{err} \\cdot k\\,.
\\end{align}
```

Where:

  - ``t_{te}``: Auxiliary tracking error scalar variable.
  - ``\\mathbf{X}``: Asset returns matrix (``T \\times N``).
  - $(math_dict[:w_port])
  - ``\\boldsymbol{b}``: Benchmark return vector.
  - $(math_dict[:k_budget])
  - ``p``: Norm order.
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
    t_te = model[Symbol(:t_te_, i)] = JuMP.@variable(model)
    tr = model[Symbol(:te_, i)] = JuMP.@expression(model, net_X - wb * k)
    model[Symbol(:cte_noc_, i)], model[Symbol(:cte_, i)] = JuMP.@constraints(model,
                                                                             begin
                                                                                 [sc * t_te;
                                                                                  sc * tr] in
                                                                                 JuMP.MOI.NormOneCone(1 +
                                                                                                      T)
                                                                                 sc *
                                                                                 (t_te -
                                                                                  f * k) <=
                                                                                 0
                                                                             end)
    return nothing
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
    f = err * sqrt(size(X, 1) - tr.alg.ddof)
    t_te = model[Symbol(:t_te_, i)] = JuMP.@variable(model)
    tr = model[Symbol(:te_, i)] = JuMP.@expression(model, net_X - wb * k)
    model[Symbol(:cte_soc_, i)], model[Symbol(:cte_, i)] = JuMP.@constraints(model,
                                                                             begin
                                                                                 [sc * t_te;
                                                                                  sc * tr] in
                                                                                 JuMP.SecondOrderCone()
                                                                                 sc *
                                                                                 (t_te -
                                                                                  f * k) <=
                                                                                 0
                                                                             end)
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
    t_te, r_te = model[Symbol(:t_te_, i)], model[Symbol(:r_te_, i)] = JuMP.@variables(model,
                                                                                      begin
                                                                                          ()
                                                                                          [1:T]
                                                                                      end)
    tr = model[Symbol(:te_, i)] = JuMP.@expression(model, net_X - wb * k)
    model[Symbol(:cte_pnorm_, i)], model[Symbol(:cste_, i)], model[Symbol(:cte_, i)] = JuMP.@constraints(model,
                                                                                                         begin
                                                                                                             [i = 1:T],
                                                                                                             [sc *
                                                                                                              r_te[i],
                                                                                                              sc *
                                                                                                              t_te,
                                                                                                              sc *
                                                                                                              tr[i]] in
                                                                                                             JuMP.MOI.PowerCone(p_inv)
                                                                                                             sc *
                                                                                                             (sum(r_te) -
                                                                                                              t_te) ==
                                                                                                             0
                                                                                                             sc *
                                                                                                             (t_te -
                                                                                                              f *
                                                                                                              k) <=
                                                                                                             0
                                                                                                         end)
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
    t_te = model[Symbol(:t_te_, i)] = JuMP.@variable(model)
    tr = model[Symbol(:te_, i)] = JuMP.@expression(model, net_X - wb * k)
    model[Symbol(:cte_infnorm_, i)], model[Symbol(:cte_, i)] = JuMP.@constraints(model,
                                                                                 begin
                                                                                     [sc *
                                                                                      t_te
                                                                                      sc *
                                                                                      tr] in
                                                                                     JuMP.MOI.NormInfinityCone(1 +
                                                                                                               T)
                                                                                     sc *
                                                                                     (t_te -
                                                                                      f * k) <=
                                                                                     0
                                                                                 end)
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
    tprefix = Symbol(prefix, :te_ir_, i, :_)
    preg!(model, tprefix, :w, JuMP.@expression(model, w - wb * k))
    risk_expr = set_risk_tracking_risk_constraints!(model, r, opt, pr, pl, fees, tprefix,
                                                    args...; kwargs...)
    model[Symbol(:cter_, i)] = JuMP.@constraint(model, sc * (risk_expr - err * k) <= 0)
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
    key = Symbol(:te_dr_, i)
    te_dr = model[key] = JuMP.@variable(model)
    tprefix = Symbol(prefix, :te_dr_, i, :_)
    preg!(model, tprefix, :w, get_w(model, prefix))
    risk_expr = set_risk_tracking_risk_constraints!(model, ri, opt, pr, pl, fees, tprefix,
                                                    args...; kwargs...)
    dr = model[Symbol(key, i)] = JuMP.@expression(model, risk_expr - rb * k)
    model[Symbol(:cter_noc_, i)], model[Symbol(:cter_, i)] = JuMP.@constraints(model,
                                                                               begin
                                                                                   [sc *
                                                                                    te_dr
                                                                                    sc * dr] in
                                                                                   JuMP.MOI.NormOneCone(2)
                                                                                   sc *
                                                                                   (te_dr -
                                                                                    err * k) <=
                                                                                   0
                                                                               end)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, pr::AbstractPriorResult,
                                         tres::Tr_VecTr, args...; kwargs...)
    for (i, tr) in enumerate(tres)
        set_tracking_error_constraints!(model, i, pr, tr, args...; kwargs...)
    end
    return nothing
end
