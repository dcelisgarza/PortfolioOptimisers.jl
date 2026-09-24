"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add tracking error constraints to the JuMP optimisation model.

The fall-through method does nothing. Concrete methods dispatch on the tracking algorithm type:

  - [`L1Norm`](@ref): Enforces `‖net_X - wb * k‖₁ ≤ err * (T - ddof)` via NormOneCone.
  - [`L2Norm`](@ref): Enforces a scaled L2 norm via SecondOrderCone.
  - [`SquaredL2Norm`](@ref): The same cone, with the bound square-rooted, because `err` bounds the *squared* error that [`norm_error`](@ref) reports.
  - [`LpNorm`](@ref): Enforces a scaled Lp norm via power cone.
  - [`LInfNorm`](@ref): Enforces `‖net_X - wb * k‖_∞ ≤ err` via NormInfinityCone.
  - [`IndependentVariableTracking`](@ref): Substitutes `w - wb` for `w` and applies the chosen risk constraint.
  - [`DependentVariableTracking`](@ref): Constrains the absolute difference between portfolio risk and benchmark risk. The inner build reads the lifted matrix of the head's weights, [`weights_prefix`](@ref). When the inner measure builds on that matrix, the bound holds from above only, as [`DependentVariableTracking`](@ref) states.

The collection method iterates over all tracking errors in `tres`.

# Mathematical definition

```math
\\begin{align}
t_{tr} &\\geq \\lVert \\mathbf{X}\\boldsymbol{w} - \\boldsymbol{b} k \\rVert_p \\cdot c_p^{-1}\\,, \\\\
t_{tr} &\\leq \\mathrm{err} \\cdot k\\,.
\\end{align}
```

Where:

  - ``t_{tr}``: Auxiliary tracking error scalar variable.
  - ``\\mathbf{X}``: Asset returns matrix (``T \\times N``).
  - $(math_dict[:w_port])
  - ``\\boldsymbol{b}``: Benchmark return vector.
  - $(math_dict[:k_budget])
  - $(math_dict[:p_norm_order])
  - ``c_p``: Norm-specific scaling factor, [`norm_factor`](@ref) of the norm (``T - d``, ``\\sqrt{T - d}``, ``1``, etc.).
  - ``\\mathrm{err}``: Tracking error tolerance.

# Arguments

  - $(arg_dict[:model])
  - `i::Integer`: Constraint index for generating unique variable and constraint names.
  - `pr::AbstractPriorResult`: Prior result providing the return matrix `X`.
  - `tr`: Tracking error specification.
  - `opt`: The [`RiskConstraintOwner`](@ref) whose programme the constraint joins, read by the risk-based variants for the solver a Deferred Quantity is resolved against ([`risk_constraint_solver`](@ref)).
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])
  - `prefix::Symbol`: The Model State namespace the entries are registered under, `Symbol("")` for a head's own, so a programme Allocation Set's tracking error joins a leader's model beside the head's own under `:aset_`. The net returns `X w` are formed under the same prefix, over the rows of `pr` and their count.

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
                                         prefix::Symbol = Symbol(""), kwargs...)
    X = pr.X
    k = get_k(model)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    wb = tracking_benchmark(tr.tr, X)
    err = tr.err
    T = size(X, 1)
    f = err * norm_factor(tr.alg, T)
    t_tr = state_set!(model, prefix, :t_tr_, i, JuMP.@variable(model))
    tr = state_set!(model, prefix, :tr_, i, JuMP.@expression(model, net_X - wb * k))
    ctr_noc, ctr = JuMP.@constraints(model,
                                     begin
                                         [sc * t_tr;
                                          sc * tr] in JuMP.MOI.NormOneCone(1 + T)
                                         sc * (t_tr - f * k) <= 0
                                     end)
    state_set!(model, prefix, :ctr_noc_, i, ctr_noc)
    state_set!(model, prefix, :ctr_, i, ctr)
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
    return err * norm_factor(f, T)
end
function tracking_error_soc_factor(f::SquaredL2Norm, err::Number, T::Integer)
    return sqrt(err * norm_factor(f, T))
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::TrackingError{<:Any, <:Any,
                                                           <:Union{<:L2Norm,
                                                                   <:SquaredL2Norm}},
                                         args...; prefix::Symbol = Symbol(""), kwargs...)
    X = pr.X
    k = get_k(model)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    wb = tracking_benchmark(tr.tr, X)
    err = tr.err
    f = tracking_error_soc_factor(tr.alg, err, size(X, 1))
    t_tr = state_set!(model, prefix, :t_tr_, i, JuMP.@variable(model))
    tr = state_set!(model, prefix, :tr_, i, JuMP.@expression(model, net_X - wb * k))
    ctr_soc, ctr = JuMP.@constraints(model,
                                     begin
                                         [sc * t_tr;
                                          sc * tr] in JuMP.SecondOrderCone()
                                         sc * (t_tr - f * k) <= 0
                                     end)
    state_set!(model, prefix, :ctr_soc_, i, ctr_soc)
    state_set!(model, prefix, :ctr_, i, ctr)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::TrackingError{<:Any, <:Any, <:LpNorm}, args...;
                                         prefix::Symbol = Symbol(""), kwargs...)
    @argcheck(tr.alg.p > 1,
              DomainError(tr.alg.p,
                          "`LpNorm.p` is $(tr.alg.p), and the tracking error is the `p`-norm of the deviation, which the model states with a power cone of exponent `1 / p`, so `1 < p` must hold. State a value greater than `1`."))
    X = pr.X
    k = get_k(model)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    wb = tracking_benchmark(tr.tr, X)
    T = size(X, 1)
    err = tr.err
    p_inv = inv(tr.alg.p)
    f = err * norm_factor(tr.alg, T)
    t_tr, r_tr = JuMP.@variables(model, begin
                                     ()
                                     [1:T]
                                 end)
    state_set!(model, prefix, :t_tr_, i, t_tr)
    state_set!(model, prefix, :r_tr_, i, r_tr)
    tr = state_set!(model, prefix, :tr_, i, JuMP.@expression(model, net_X - wb * k))
    ctr_pnorm, cstr, ctr = JuMP.@constraints(model,
                                             begin
                                                 [i = 1:T],
                                                 [sc * r_tr[i], sc * t_tr, sc * tr[i]] in
                                                 JuMP.MOI.PowerCone(p_inv)
                                                 sc * (sum(r_tr) - t_tr) == 0
                                                 sc * (t_tr - f * k) <= 0
                                             end)
    state_set!(model, prefix, :ctr_pnorm_, i, ctr_pnorm)
    state_set!(model, prefix, :cstr_, i, cstr)
    state_set!(model, prefix, :ctr_, i, ctr)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::TrackingError{<:Any, <:Any, <:LInfNorm},
                                         args...; prefix::Symbol = Symbol(""), kwargs...)
    X = pr.X
    k = get_k(model)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    wb = tracking_benchmark(tr.tr, X)
    T = size(X, 1)
    err = tr.err
    f = err * norm_factor(tr.alg, T)
    t_tr = state_set!(model, prefix, :t_tr_, i, JuMP.@variable(model))
    tr = state_set!(model, prefix, :tr_, i, JuMP.@expression(model, net_X - wb * k))
    ctr_infnorm, ctr = JuMP.@constraints(model,
                                         begin
                                             [sc * t_tr
                                              sc * tr] in JuMP.MOI.NormInfinityCone(1 + T)
                                             sc * (t_tr - f * k) <= 0
                                         end)
    state_set!(model, prefix, :ctr_infnorm_, i, ctr_infnorm)
    state_set!(model, prefix, :ctr_, i, ctr)
    return nothing
end
function set_tracking_error_constraints!(model::JuMP.Model, i::Integer,
                                         pr::AbstractPriorResult,
                                         tr::RiskTrackingError{<:Any, <:Any, <:Any,
                                                               <:IndependentVariableTracking},
                                         opt::Union{<:JuMPOptimisationEstimator,
                                                    <:AbstractProgrammeAllocationSet},
                                         pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees},
                                         args...; prefix::Symbol = Symbol(""), kwargs...)
    r = tr.r
    wb = tr.tr.w
    err = tr.err
    w = get_w(model, prefix)
    k = get_k(model)
    sc = get_constraint_scale(model)
    tprefix = nested_prefix(prefix, :tr_ir_, i)
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
                                         opt::Union{<:JuMPOptimisationEstimator,
                                                    <:AbstractProgrammeAllocationSet},
                                         pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees},
                                         args...; prefix::Symbol = Symbol(""), kwargs...)
    ri = tr.r
    wb = tr.tr.w
    err = tr.err
    rb = expected_risk(factory(ri, pr, risk_constraint_solver(opt)), wb, pr.X, fees)
    k = get_k(model)
    sc = get_constraint_scale(model)
    tr_dr = state_set!(model, prefix, :tr_dr_, i, JuMP.@variable(model))
    tprefix = nested_prefix(prefix, :tr_dr_, i)
    state_set!(model, tprefix, :w, get_w(model, prefix))
    state_set!(model, tprefix, :w_owner, weights_prefix(model, prefix))
    risk_expr = set_risk_tracking_risk_constraints!(model, ri, opt, pr, pl, fees, tprefix,
                                                    args...; kwargs...)
    # The risk difference is its own entry name. It was `Symbol(key, i)` — the *composed*
    # key with the index appended a second time — which put entry 1 at `:tr_dr_11`, the
    # key `tr_dr` itself takes at entry 11.
    dr = state_set!(model, prefix, :tr_dr_diff_, i,
                    JuMP.@expression(model, risk_expr - rb * k))
    cter_noc, cter = JuMP.@constraints(model,
                                       begin
                                           [sc * tr_dr
                                            sc * dr] in JuMP.MOI.NormOneCone(2)
                                           sc * (tr_dr - err * k) <= 0
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
