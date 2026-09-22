"""
    calibrated(x)
    calibrated(x::AbstractVector)

Whether a penalty or a norm-ceiling slot holds a Calibration Rule, which resolves against the prior result and so reads the head's rows: the slot itself, or a slot of the term it holds ([`calibration_slots`](@ref)).

# Related

  - [`rows_needed`](@ref)
  - [`resolve_allocation_set_rows`](@ref)
  - [`is_calibration_rule`](@ref)
"""
function calibrated(x::Any)
    return is_calibration_rule(x) || any(is_calibration_rule, values(calibration_slots(x)))
end
function calibrated(x::AbstractVector)
    return any(calibrated, x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The second stage of a [`ProgrammeAllocationSet`](@ref)'s resolution, at every step on the prior result `pr` fitted on the head's rows `X`, or on `nothing` when the set reads no rows: an exposure row in `lcse` re-based through the prior's loadings, the centrality and phylogeny estimators fitted on the rows, the return floor materialised against the prior, and every Calibration Rule in a norm ceiling or a penalty resolved, as [`processed_jump_optimiser_attributes`](@ref) and [`assemble_jump_model!`](@ref) resolve them. The head's `strict` and its pinned names come off the current [`ProjectionStep`](@ref), the rows as the carrier [`leader_carrier`](@ref) forms.

# Returns

  - `attrs::NamedTuple`: `lcsr`, `ctr`, `plr`, `ret`, `l2c`, `lpc`, `linfc`, `l1`, `l2`, `lp`, `linf` and the `ReturnsResult` `rd` over the rows, `nothing` when there are none.

# Related

  - [`resolve_allocation_set`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`linear_constraints`](@ref)
  - [`centrality_constraints`](@ref)
  - [`phylogeny_constraints`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function resolve_allocation_set_rows(set::ProgrammeAllocationSet, ::Nothing, ::Any)
    return (; lcsr = set.lcse, ctr = set.cte, plr = set.ple, ret = set.ret, l2c = set.l2c,
            lpc = set.lpc, linfc = set.linfc, l1 = set.l1, l2 = set.l2, lp = set.lp,
            linf = set.linf, rd = nothing)
end
function resolve_allocation_set_rows(set::ProgrammeAllocationSet, pr::AbstractPriorResult,
                                     X::AbstractMatrix)
    rd = leader_carrier(X)
    strict = projection_step_strict()
    slv = set.slv
    lcsr = if exposure_keyed(set.lcse)
        linear_constraints(set.lcse, set.sets; datatype = eltype(X), strict = strict,
                           rr = pr.rr, rd = rd)
    else
        set.lcse
    end
    ctr = centrality_constraints(set.cte, pr; rd = rd, strict = strict)
    plr = phylogeny_constraints(set.ple, pr; rd = rd)
    ret = factory(set.ret, pr)
    l2c = resolve_calibration_slot(set.l2c, :l2c, pr, pr.w, slv,
                                   CalibrationContext(; p = 2))
    linfc = resolve_calibration_slot(set.linfc, :linfc, pr, pr.w, slv,
                                     CalibrationContext(; p = Inf))
    lpc = norm_ceiling_factory(set.lpc, pr, slv)
    l1 = resolve_calibration_slot(set.l1, :l1, pr, pr.w, slv)
    l2 = factory(set.l2, pr, slv)
    lp = factory(set.lp, pr, slv)
    linf = resolve_calibration_slot(set.linf, :linf, pr, pr.w, slv)
    return (; lcsr, ctr, plr, ret, l2c, lpc, linfc, l1, l2, lp, linf, rd)
end
"""
    set_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet, w::AbstractVector, X)
    set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X)

Adds the constraints of a resolved Allocation Set to a model that already carries `w`, `k`, the scales and the observation count: the budget one and the weight bounds on both kinds, and on the programme set every kind through the shared builders, in the order [`assemble_jump_model!`](@ref) runs them — the linear and centrality rows, the MIP kinds, the sub-group MIP kinds, the turnover ceilings with their reference replaced by `w`, the tracking errors, the norm ceilings, the penalties into the Objective Penalty, the risk ceilings, the return floor, the integer and semidefinite phylogeny rows, and the custom constraints — the second-stage resolution ([`resolve_allocation_set_rows`](@ref)) run first on the prior fitted on `X`.

The bare projection model of [`project`](@ref) is one caller; a JuMP head that takes the set as its programme's feasible region is another, through [`add_allocation_set_constraints!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `set`: The Allocation Set, resolved once per fold.
  - `w`: The Price-Adjusted Allocation the step trades from, the reference of the turnover ceilings and of a tracking benchmark that is not fixed.
  - `X`: The rows of returns the head holds through the period, `observations × assets`, or `nothing`.

# Validation

  - A slot that reads the rows with `X === nothing`. An `ArgumentError` is thrown: the rows reach a projection inside an Online Update alone.

# Returns

  - `nothing`.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`add_allocation_set_constraints!`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`set_allocation_risk_ceiling!`](@ref)
"""
function set_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet,
                                         ::AbstractVector, ::Any)::Nothing
    set_allocation_set_bounds!(model, set.wb)
    return nothing
end
function set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                         w::AbstractVector, X)::Nothing
    set_allocation_set_bounds!(model, set.wb, set.sbgt, set.gbgt)
    assemble_allocation_set!(model, set, w, X, Symbol(""))
    return nothing
end
"""
    add_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet, w::AbstractVector, X)
    add_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X)

Adds the constraints of a resolved Allocation Set to a JuMP head's model mid-assembly, the arm of the Allocation Set Constraint a [`FollowTheLeader`](@ref) rule appends to its held optimiser.

The head's own builders have already registered the model's named entries — its weight bounds, its budgets, its turnover and tracking-error terms under their indices — so this arm adds the set's rows without a name where the bare projection model names them: the bounds, the budget of one and the short and gross budgets as anonymous constraints, and every other kind through the same builders as [`set_allocation_set_constraints!`](@ref), in the same order, under the `:aset_` prefix where a builder takes one, at the first index the model has not used where it takes an index, and with the head's `w` registered under the prefix so a ceiling's entries never meet the head's own measures'. The set's penalties and its semidefinite phylogeny's `p · tr(W)` fold into the Objective Penalty the head's objective builder folds in, so the adapter is one door. The tracking errors are written over the head's rows `X` and their count, not the selection's.

A kind whose builder names its entries once per model — a MIP kind, the exact long-short pin, a norm ceiling, a penalty, an integer phylogeny — is stated in one home: on the set, or on the held optimiser. Stated in both, the second registration fails by name.

# Arguments

  - $(arg_dict[:model])
  - `set`: The Allocation Set, resolved once per fold.
  - `w`: The Price-Adjusted Allocation the step trades from.
  - `X`: The rows of returns the head holds through the period, `observations × assets`, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_allocation_set_constraints!`](@ref)
  - [`AllocationSetConstraint`](@ref)
  - [`FollowTheLeader`](@ref)
"""
function add_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet,
                                         ::AbstractVector, ::Any)::Nothing
    add_allocation_set_bounds!(model, set.wb, nothing, nothing)
    return nothing
end
function add_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                         w::AbstractVector, X)::Nothing
    add_allocation_set_bounds!(model, set.wb, set.sbgt, set.gbgt)
    assemble_allocation_set!(model, set, w, X, :aset_)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The builder sequence both arms share, after the bounds: every kind of a programme set in the order [`assemble_jump_model!`](@ref) runs them, under `prefix` — `Symbol("")` on the bare projection model, `:aset_` in a leader's model, where the indexed builders take the first free index and the head's `w` is registered under the prefix.

# Related

  - [`set_allocation_set_constraints!`](@ref)
  - [`add_allocation_set_constraints!`](@ref)
  - [`resolve_allocation_set_rows`](@ref)
"""
function assemble_allocation_set!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                  w::AbstractVector, X, prefix::Symbol)::Nothing
    bare = prefix == Symbol("")
    pr = allocation_set_prior(set, X)
    (; lcsr, ctr, plr, ret, l2c, lpc, linfc, l1, l2, lp, linf, rd) = resolve_allocation_set_rows(set,
                                                                                                 pr,
                                                                                                 X)
    if !bare
        state_build!(() -> get_w(model), model, prefix, :w)
    end
    set_linear_weight_constraints!(model, lcsr, state_key(prefix, :lcs_ineq_),
                                   state_key(prefix, :lcs_eq_))
    set_linear_weight_constraints!(model, ctr, state_key(prefix, :cent_ineq_),
                                   state_key(prefix, :cent_eq_))
    set_mip_constraints!(model, set.wb, set.card, set.gcarde, plr, set.lt, set.st, nothing,
                         set.ss, set.xbgt)
    set_smip_constraints!(model, set.wb, set.scard, set.sgcarde, set.smtx, set.sgmtx,
                          set.slt, set.sst, set.sglt, set.sgst, set.ss)
    set_allocation_turnover!(model, factory(set.tn, w), bare)
    if !isnothing(set.tr)
        tr = prefix_tracking_benchmark(factory(set.tr, w), size(pr.X, 1))
        set_tracking_error_constraints!(model, pr, tr, set, plr, nothing; prefix = prefix)
    end
    set_weight_norm_2_constraints!(model, l2c)
    set_weight_norm_p_constraints!(model, lpc)
    set_weight_norm_inf_constraints!(model, linfc)
    set_l1_regularisation!(model, l1)
    set_l2_regularisation!(model, l2)
    set_lp_regularisation!(model, lp)
    set_linf_regularisation!(model, linf)
    set_allocation_risk_ceiling!(model, set.r, set, pr, plr; prefix = prefix)
    set_allocation_return_floor!(model, ret, pr, bare; rd = rd)
    set_iplg_constraints!(model, plr)
    set_sdp_phylogeny_constraints!(model, plr; prefix = prefix)
    add_custom_constraint!(model, set.ccnt, set, nothing)
    return nothing
end
"""
    prefix_tracking_benchmark(tr, T::Integer)
    prefix_tracking_benchmark(tr::TrackingError{<:ReturnsTracking}, T::Integer)
    prefix_tracking_benchmark(trs::AbstractVector, T::Integer)

A [`ReturnsTracking`](@ref) benchmark on a programme set, cut to the first `T` rows when it is longer: the head's rows are the prefix of the fold's rows it has folded so far, so a benchmark series stated over the fold aligns with them by its prefix. A benchmark shorter than the rows reaches the builder as it is and is refused there by length. A weights benchmark reads the rows themselves and needs no cut.

# Related

  - [`assemble_allocation_set!`](@ref)
  - [`set_tracking_error_constraints!`](@ref)
  - [`ReturnsTracking`](@ref)
"""
function prefix_tracking_benchmark(tr::Any, ::Integer)
    return tr
end
function prefix_tracking_benchmark(tr::TrackingError{<:ReturnsTracking}, T::Integer)
    b = tr.tr.w
    if length(b) <= T
        return tr
    end
    return TrackingError(; tr = ReturnsTracking(; w = b[1:T]), err = tr.err, alg = tr.alg)
end
function prefix_tracking_benchmark(trs::AbstractVector, T::Integer)
    return [prefix_tracking_benchmark(tr, T) for tr in trs]
end
"""
    set_allocation_turnover!(model::JuMP.Model, tn::Nothing, bare::Bool)
    set_allocation_turnover!(model::JuMP.Model, tn::Turnover, bare::Bool)
    set_allocation_turnover!(model::JuMP.Model, tns::VecTn, bare::Bool)

Writes a programme set's turnover ceilings, their reference already replaced by the step's Price-Adjusted Allocation: at the model's own indices on the bare projection model, and at the first indices the model has not used in a leader's model, so the set's terms sit beside the head's own.

# Related

  - [`set_turnover_constraints!`](@ref)
  - [`free_state_index`](@ref)
"""
function set_allocation_turnover!(::JuMP.Model, ::Nothing, ::Bool)::Nothing
    return nothing
end
function set_allocation_turnover!(model::JuMP.Model, tn::Turnover, bare::Bool)::Nothing
    set_allocation_turnover!(model, [tn], bare)
    return nothing
end
function set_allocation_turnover!(model::JuMP.Model, tns::VecTn, bare::Bool)::Nothing
    if bare
        set_turnover_constraints!(model, tns)
        return nothing
    end
    for tn in tns
        _set_turnover_constraints!(model, tn, free_state_index(model, :t_tn_))
    end
    return nothing
end
"""
    set_allocation_return_floor!(model::JuMP.Model, ret::Nothing, pr, bare::Bool; kwargs...)
    set_allocation_return_floor!(model::JuMP.Model, ret::JuMPReturnsEstimator, pr::AbstractPriorResult, bare::Bool; kwargs...)
    set_allocation_return_floor!(model::JuMP.Model, rets::VecJRE, pr::AbstractPriorResult, bare::Bool; kwargs...)

Writes a programme set's return floor: each term's return expression on the prior's expected returns and its `settings.lb` row, through the per-index arm of [`set_return_constraints!`](@ref), at the model's own indices on the bare projection model and at the first indices the model has not used in a leader's model. The term's objective role — its scalarised expression, a ratio's normalisation — is not read: the set's objective is the geometry's divergence.

# Related

  - [`set_return_constraints!`](@ref)
  - [`free_state_index`](@ref)
"""
function set_allocation_return_floor!(::JuMP.Model, ::Nothing, ::Any, ::Bool;
                                      kwargs...)::Nothing
    return nothing
end
function set_allocation_return_floor!(model::JuMP.Model, ret::JuMPReturnsEstimator,
                                      pr::AbstractPriorResult, bare::Bool;
                                      kwargs...)::Nothing
    set_allocation_return_floor!(model, [ret], pr, bare; kwargs...)
    return nothing
end
function set_allocation_return_floor!(model::JuMP.Model, rets::VecJRE,
                                      pr::AbstractPriorResult, bare::Bool;
                                      kwargs...)::Nothing
    for (i, ret) in enumerate(rets)
        j = bare ? i : free_state_index(model, :ret_)
        set_return_constraints!(model, j, unit_scale_returns_estimator(ret), pr; kwargs...)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds an Allocation Set's weight bounds, its budget of one and its short and gross budgets to a model that already names its own: the same inequalities [`set_allocation_set_bounds!`](@ref) writes, as anonymous constraints, the budgets over the model's own long and short parts when it holds them and over anonymous parts bound to `w` otherwise.

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub` over the resolved bounds, through [`assert_feasible_bounds`](@ref).

# Related

  - [`add_allocation_set_constraints!`](@ref)
  - [`set_allocation_set_bounds!`](@ref)
  - [`anonymous_budget_rows!`](@ref)
"""
function add_allocation_set_bounds!(model::JuMP.Model, wb::WeightBounds, sbgt,
                                    gbgt)::Nothing
    assert_feasible_bounds(wb)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    if w_finite_flag(wb.lb)
        JuMP.@constraint(model, sc * (w ⊖ k * wb.lb) >= 0)
    end
    if w_finite_flag(wb.ub)
        JuMP.@constraint(model, sc * (w ⊖ k * wb.ub) <= 0)
    end
    JuMP.@constraint(model, sc * (sum(w) - k) == 0)
    if isnothing(sbgt) && isnothing(gbgt)
        return nothing
    end
    lw, sw = if shared_has(model, :sw)
        shared_get(model, :lw), shared_get(model, :sw)
    else
        N = length(w)
        lw = JuMP.@variable(model, [1:N], lower_bound = 0)
        sw = JuMP.@variable(model, [1:N], lower_bound = 0)
        JuMP.@constraint(model, sc * (w - lw) <= 0)
        JuMP.@constraint(model, sc * (w + sw) >= 0)
        lw, sw
    end
    anonymous_budget_rows!(model, sum(sw), sbgt)
    anonymous_budget_rows!(model, sum(lw) + sum(sw), gbgt)
    return nothing
end
"""
    anonymous_budget_rows!(model::JuMP.Model, expr, bgt::Nothing)
    anonymous_budget_rows!(model::JuMP.Model, expr, bgt::Number)
    anonymous_budget_rows!(model::JuMP.Model, expr, bgt::BudgetRange)

One budget on an expression as anonymous rows: pinned to `k · bgt` by a number, bounded by a [`BudgetRange`](@ref), none for `nothing`.

# Related

  - [`add_allocation_set_bounds!`](@ref)
  - [`set_budget_constraints!`](@ref)
"""
function anonymous_budget_rows!(::JuMP.Model, ::Any, ::Nothing)::Nothing
    return nothing
end
function anonymous_budget_rows!(model::JuMP.Model, expr, bgt::Number)::Nothing
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, sc * (expr - k * bgt) == 0)
    return nothing
end
function anonymous_budget_rows!(model::JuMP.Model, expr, bgt::BudgetRange)::Nothing
    k = get_k(model)
    sc = get_constraint_scale(model)
    if !isnothing(bgt.lb)
        JuMP.@constraint(model, sc * (expr - k * bgt.lb) >= 0)
    end
    if !isnothing(bgt.ub)
        JuMP.@constraint(model, sc * (expr - k * bgt.ub) <= 0)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The first index `i` at which the Model State entry `name` is not registered, so a builder keyed by an index can add its term beside a head's own.

# Related

  - [`add_allocation_set_constraints!`](@ref)
  - [`state_set!`](@ref)
"""
function free_state_index(model::JuMP.Model, name::Symbol)
    i = 1
    while state_has(model, Symbol(""), name, i)
        i += 1
    end
    return i
end
"""
    set_allocation_set_bounds!(model::JuMP.Model, wb::WeightBounds, sbgt = nothing, gbgt = nothing)

Adds an Allocation Set's weight bounds, its budget of one and its short and gross budgets to the model: the bounds through [`set_weight_constraints!`](@ref) with no net budget, so a negative lower bound builds the long-short decomposition without pinning its long side — under [`JuMPOptimiser`](@ref) a net budget with no short budget pins the long side to it, which forbids every short, and an Allocation Set admits a long-short book — the short budget on its short side and the gross budget on its leverage through the same builder, and then `Σw = k` through [`set_budget_constraints!`](@ref).

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub` over the resolved bounds, through [`assert_feasible_bounds`](@ref).

# Related

  - [`set_allocation_set_constraints!`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
"""
function set_allocation_set_bounds!(model::JuMP.Model, wb::WeightBounds, sbgt = nothing,
                                    gbgt = nothing)::Nothing
    assert_feasible_bounds(wb)
    set_weight_constraints!(model, wb, nothing, sbgt; gbgt = gbgt)
    set_budget_constraints!(model, 1, get_w(model))
    return nothing
end
"""
    set_projection_objective!(model::JuMP.Model, proj::EuclideanProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::EntropicProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::GramProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::TsallisProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::LogBarrierProjection, q::AbstractVector)

Sets the projection programme's objective in the geometry's divergence from the raw step `q`: the Euclidean distance, or half its square when a penalty joins it ([`set_distance_cone!`](@ref)), the relative entropy ``\\sum_i w_i \\log (w_i / q_i)`` through a relative-entropy cone over the positive entries of `q` with the zero entries pinned at zero, the Gram norm ``\\lVert G (\\boldsymbol{w} - \\boldsymbol{q}) \\rVert`` with ``G^\\intercal G = A``, or half its square under a penalty, through the same cones, the Tsallis divergence ``-\\sum_i w_i^\\alpha / \\alpha + \\sum_i q_i^{\\alpha - 1} w_i`` through one power cone per positive entry, and the Itakura–Saito divergence ``-\\sum_i \\log w_i + \\sum_i w_i / q_i`` through one exponential cone per positive entry; under both barriers a zero entry of `q` is pinned at zero, as under the entropic arm.

The Euclidean and Gram arms minimise the norm when the objective is the divergence alone and half its square when a penalty joins it: the two have the same minimiser on their own and the norm's is the sharper for a solver, but a penalty added to the norm cannot move the minimiser off `q` until its gradient exceeds one, whereas added to the squared norm it is the proximal step the theory names. Both cones keep the programme conic on every solver. The two barrier arms drop the terms constant in `w`, so their objective value is the divergence up to a constant, with the same minimiser. Every arm hands its divergence to [`set_divergence_objective!`](@ref), which folds in the set's penalties.

# Validation

  - Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref): `all(>= 0, q)` and `sum(q) > 0`. A `DomainError` is thrown otherwise.
  - Under [`GramProjection`](@ref): `proj.A` is not `nothing`. An `ArgumentError` is thrown otherwise: the rule binds the matrix.

# Related

  - [`project`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
"""
function set_projection_objective!(model::JuMP.Model, ::EuclideanProjection,
                                   q::AbstractVector)::Nothing
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t_proj)
    set_distance_cone!(model, t_proj, sc * (w - q), sc)
    set_divergence_objective!(model, t_proj, so)
    return nothing
end
function set_projection_objective!(model::JuMP.Model, ::EntropicProjection,
                                   q::AbstractVector)::Nothing
    @argcheck(all(x -> x >= zero(x), q),
              DomainError(q,
                          "the entropic projection is defined on non-negative raw steps alone: `log w` is undefined below zero"))
    @argcheck(sum(q) > zero(eltype(q)),
              DomainError(q,
                          "the entropic projection needs a positive entry to normalise: a raw step of zeros has no projection onto the simplex"))
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    pos = findall(x -> x > zero(x), q)
    zer = findall(iszero, q)
    JuMP.@variable(model, t_proj)
    # `w_i log(w_i / q_i)` with `q_i = 0` is finite at `w_i = 0` alone: a zero entry stays
    # zero, as it does under the closed form.
    if !isempty(zer)
        JuMP.@constraint(model, proj_zero, sc * w[zer] .== 0)
    end
    JuMP.@constraint(model, proj_rec,
                     [sc * t_proj; sc * q[pos]; sc * w[pos]] in
                     JuMP.MOI.RelativeEntropyCone(1 + 2 * length(pos)))
    set_divergence_objective!(model, t_proj, so)
    return nothing
end
function set_projection_objective!(model::JuMP.Model, proj::GramProjection,
                                   q::AbstractVector)::Nothing
    @argcheck(!isnothing(proj.A),
              ArgumentError("a GramProjection projects in the norm of a Gram matrix the rule binds at each step, and this one has none bound: it is the rule's geometry, not a caller's."))
    G = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(proj.A)).U
    w = get_w(model)
    sc = get_constraint_scale(model)
    so = get_objective_scale(model)
    JuMP.@variable(model, t_proj)
    set_distance_cone!(model, t_proj, sc * (G * (w - q)), sc)
    set_divergence_objective!(model, t_proj, so)
    return nothing
end
function set_projection_objective!(model::JuMP.Model, proj::TsallisProjection,
                                   q::AbstractVector)::Nothing
    pos, w, sc, so = barrier_objective_entries(model, q)
    a = proj.alpha
    JuMP.@variable(model, t_proj[1:length(pos)])
    # `t_i ≤ w_i^α` is the power cone `w_i^α · 1^(1 − α) ≥ |t_i|`.
    for (i, j) in enumerate(pos)
        JuMP.@constraint(model, [sc * w[j], sc, sc * t_proj[i]] in JuMP.MOI.PowerCone(a))
    end
    set_divergence_objective!(model,
                              LinearAlgebra.dot(q[pos] .^ (a - 1), w[pos]) -
                              sum(t_proj) / a, so)
    return nothing
end
function set_projection_objective!(model::JuMP.Model, ::LogBarrierProjection,
                                   q::AbstractVector)::Nothing
    pos, w, sc, so = barrier_objective_entries(model, q)
    JuMP.@variable(model, t_proj[1:length(pos)])
    # `t_i ≤ log w_i` is the exponential cone `1 · exp(t_i) ≤ w_i`.
    for (i, j) in enumerate(pos)
        JuMP.@constraint(model,
                         [sc * t_proj[i], sc, sc * w[j]] in JuMP.MOI.ExponentialCone())
    end
    set_divergence_objective!(model, LinearAlgebra.dot(inv.(q[pos]), w[pos]) - sum(t_proj),
                              so)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The cone that bounds a projection's distance variable `t` by the scaled deviation `x`: the second-order cone `t ≥ ‖x‖` when the model holds no Objective Penalty, whose minimum is sharp and which the solver answers exactly; and the rotated second-order cone `2 t · 1 ≥ ‖x‖²`, so `t ≥ ½ ‖w − q‖²` after the scale, when a penalty was accumulated before the objective — the penalties, a custom objective term, a semidefinite phylogeny's `p · tr(W)` — because a penalty added to the norm cannot move the minimiser off `q` until its gradient exceeds one, whereas added to half the square it is the proximal step.

# Related

  - [`set_projection_objective!`](@ref)
  - [`set_divergence_objective!`](@ref)
"""
function set_distance_cone!(model::JuMP.Model, t, x, sc)::Nothing
    if shared_has(model, :op)
        JuMP.@constraint(model, proj_rsoc, [sc * t; sc; x] in JuMP.RotatedSecondOrderCone())
    else
        JuMP.@constraint(model, proj_soc, [sc * t; x] in JuMP.SecondOrderCone())
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Sets the projection programme's objective to the geometry's divergence `div` plus the Objective Penalty the set's penalties and its semidefinite phylogeny accumulated — the door every JuMP head's objective takes through [`add_penalty_to_objective!`](@ref) — scaled by `so`, minimised. With no penalty the objective is the divergence alone, the paper's projection.

# Related

  - [`set_projection_objective!`](@ref)
  - [`add_penalty_to_objective!`](@ref)
  - [`add_to_objective_penalty!`](@ref)
"""
function set_divergence_objective!(model::JuMP.Model, div, so)::Nothing
    obj_expr = JuMP.@expression(model, obj_expr, 0 + div)
    obj_expr = add_penalty_to_objective!(model, 1, obj_expr)
    JuMP.@objective(model, Min, so * obj_expr)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The entries a barrier objective is written over: refuses a raw step with a negative entry or none positive, pins every zero entry of `q` at zero, and answers the positive indices with the model's `w` and its two scales.

# Related

  - [`set_projection_objective!`](@ref)
  - [`TsallisProjection`](@ref)
  - [`LogBarrierProjection`](@ref)
"""
function barrier_objective_entries(model::JuMP.Model, q::AbstractVector)
    @argcheck(all(x -> x >= zero(x), q),
              DomainError(q,
                          "a barrier projection is defined on non-negative raw steps alone: the potential is undefined below zero"))
    @argcheck(sum(q) > zero(eltype(q)),
              DomainError(q,
                          "a barrier projection needs a positive entry: a raw step of zeros has no projection, because a zero stays zero under the potential"))
    w = get_w(model)
    sc = get_constraint_scale(model)
    for i in findall(iszero, q)
        JuMP.@constraint(model, sc * w[i] == 0)
    end
    return findall(x -> x > zero(x), q), w, sc, get_objective_scale(model)
end
"""
    projection_solver(proj::AbstractProjectionGeometry, set::ProgrammeAllocationSet)
    projection_solver(proj::GramProjection, set::AbstractAllocationSet)
    projection_solver(proj::GramProjection, set::ProgrammeAllocationSet)
    projection_solver(proj::GramProjection, set::BoundedAllocationSet)
    projection_solver(proj::AbstractProjectionGeometry, set::BoundedAllocationSet)

The solver a projection programme runs on: the geometry's own when it carries one, the set's otherwise. A scalar-root geometry on the bounded set has no programme to run and is refused by name; [`project`](@ref) never reaches it.

# Related

  - [`GramProjection`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
  - [`project`](@ref)
"""
function projection_solver(::AbstractProjectionGeometry, set::ProgrammeAllocationSet)
    return set.slv
end
function projection_solver(proj::GramProjection, ::AbstractAllocationSet)
    return proj.slv
end
function projection_solver(proj::GramProjection, ::ProgrammeAllocationSet)
    return proj.slv
end
function projection_solver(proj::GramProjection, ::BoundedAllocationSet)
    return proj.slv
end
function projection_solver(proj::AbstractProjectionGeometry, ::BoundedAllocationSet)
    return throw(ArgumentError("a `$(nameof(typeof(proj)))` onto a BoundedAllocationSet is a scalar root, not a programme: no solver is read for it."))
end
"""
    allocation_set_ready(set::AbstractAllocationSet, X)
    allocation_set_ready(set::ProgrammeAllocationSet, X)

Whether a set's constraints can be formed on the rows `X` the step holds: `nothing` when they can, and the reason for a Held Step when they cannot — a programme set that fits its prior on the rows, while the head holds fewer than two, because a covariance of one observation does not exist.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`projection_programme`](@ref)
  - [`HeldStep`](@ref)
"""
function allocation_set_ready(::AbstractAllocationSet, ::Any)
    return nothing
end
function allocation_set_ready(set::ProgrammeAllocationSet, X)
    if isnothing(rows_needed(set)) && !isnothing(X) && size(X, 1) < 2
        return "the set's prior estimator fits on the head's rows and a covariance of one observation does not exist, so the step trades nothing until the head holds two"
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The projection programme: a bare model with `w`, `k = 1`, the set's scales and the observation count of the step's rows, the set's constraints from [`set_allocation_set_constraints!`](@ref) with `w` as the turnover reference, the set's custom objective terms through [`add_custom_objective_term!`](@ref) with the geometry as the objective and the set as the owner, and the geometry's objective from [`set_projection_objective!`](@ref), solved on [`projection_solver`](@ref).

A solved programme answers its weights. A failed one, or one whose constraints cannot be formed on the step's rows ([`allocation_set_ready`](@ref)), is the Held Step: the record goes to the current [`ProjectionStep`](@ref) through [`record_held_step!`](@ref), and the answer is a copy of `w`, the book the fund already holds.

# Arguments

  - `proj`: The geometry.
  - `set`: The set, resolved.
  - `q`: The raw step.
  - `w`: The Price-Adjusted Allocation the step trades from.

# Returns

  - `w'::Vector`: The projected allocation, or `w` copied on a Held Step.

# Related

  - [`project`](@ref)
  - [`HeldStep`](@ref)
  - [`optimise_JuMP_model!`](@ref)
"""
function projection_programme(proj::AbstractProjectionGeometry, set::AbstractAllocationSet,
                              q::AbstractVector, w::AbstractVector)
    X = projection_step_rows()
    reason = allocation_set_ready(set, X)
    if !isnothing(reason)
        record_held_step!(reason, nothing)
        return copy(w)
    end
    model = JuMP.Model()
    set_model_scales!(model, projection_scale(set, :sc), projection_scale(set, :so))
    set_model_observations!(model, isnothing(X) ? 0 : size(X, 1))
    JuMP.@expression(model, k, 1)
    wv = state_set!(model, Symbol(""), :w,
                    JuMP.@variable(model, [1:length(q)], base_name = "w"))
    set_allocation_set_constraints!(model, set, w, X)
    add_custom_objective_term!(model, proj, allocation_set_cobj(set), set, nothing)
    set_projection_objective!(model, proj, q)
    res = optimise_JuMP_model!(model, projection_solver(proj, set))
    if res.success
        return JuMP.value.(wv)
    end
    record_held_step!("the projection onto the `$(nameof(typeof(set)))` in the `$(nameof(typeof(proj)))` geometry did not solve, and the step trades nothing",
                      res.trials)
    return copy(w)
end
"""
    projection_scale(set::ProgrammeAllocationSet, f::Symbol)
    projection_scale(set::BoundedAllocationSet, f::Symbol)

The constraint scale `:sc` or the objective scale `:so` a projection programme registers on its model: the programme set's own, and one on the bounded set, which carries none.

# Related

  - [`projection_programme`](@ref)
  - [`set_model_scales!`](@ref)
"""
function projection_scale(set::ProgrammeAllocationSet, f::Symbol)
    return getfield(set, f)
end
function projection_scale(::BoundedAllocationSet, ::Symbol)
    return 1
end
"""
    allocation_set_cobj(set::ProgrammeAllocationSet)
    allocation_set_cobj(set::BoundedAllocationSet)

The custom objective terms a projection programme folds into its objective: the programme set's `cobj`, and none on the bounded set, which carries no objective term.

# Related

  - [`projection_programme`](@ref)
  - [`add_custom_objective_term!`](@ref)
"""
function allocation_set_cobj(set::ProgrammeAllocationSet)
    return set.cobj
end
function allocation_set_cobj(::BoundedAllocationSet)
    return nothing
end
"""
    project(proj::GramProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::EuclideanProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::EntropicProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::GramProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::TsallisProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)
    project(proj::LogBarrierProjection, set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)

The programme arms of the Constrained Update: every pair but the scalar roots on the bounded set is the bare-model programme of [`projection_programme`](@ref).

# Validation

  - Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref): `all(>= 0, lb)` over the resolved bounds. A `DomainError` is thrown otherwise. Under the same three geometries the programme's answer passes through [`clip_at_zero`](@ref), so a leg the solver closed to within its tolerance below zero is zero to the next step.

# Related

  - [`project`](@ref)
  - [`projection_programme`](@ref)
  - [`ProgrammeAllocationSet`](@ref)
  - [`GramProjection`](@ref)
"""
function project(proj::GramProjection, set::BoundedAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    return projection_programme(proj, set, q, w)
end
function project(proj::EuclideanProjection, set::ProgrammeAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    return projection_programme(proj, set, q, w)
end
function project(proj::EntropicProjection, set::ProgrammeAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    @argcheck(all(x -> x >= zero(x), set.wb.lb),
              DomainError(set.wb.lb,
                          "the entropic projection admits no negative lower bound: `log w` is undefined below zero"))
    return clip_at_zero(projection_programme(proj, set, q, w))
end
function project(proj::GramProjection, set::ProgrammeAllocationSet, q::AbstractVector,
                 w::AbstractVector)
    return projection_programme(proj, set, q, w)
end
function project(proj::Union{<:TsallisProjection, <:LogBarrierProjection},
                 set::ProgrammeAllocationSet, q::AbstractVector, w::AbstractVector)
    @argcheck(all(x -> x >= zero(x), set.wb.lb),
              DomainError(set.wb.lb,
                          "a barrier projection admits no negative lower bound: the potential is undefined below zero"))
    return clip_at_zero(projection_programme(proj, set, q, w))
end
"""
    blend_projection(proj::EuclideanProjection, set::BoundedAllocationSet, q::AbstractVector, w::AbstractVector)
    blend_projection(proj::EuclideanProjection, set::AbstractAllocationSet, q::AbstractVector, w::AbstractVector)

The second projection of an [`ExpertMixture`](@ref)'s blend onto the head's set: skipped by dispatch on a [`BoundedAllocationSet`](@ref), where a blend of bounded allocations is bounded and the projection would be the identity, and [`project`](@ref) on every other set, where it is the repair a turnover ceiling or a MIP kind needs.

# Related

  - [`ExpertMixture`](@ref)
  - [`project`](@ref)
"""
function blend_projection(::EuclideanProjection, ::BoundedAllocationSet, q::AbstractVector,
                          ::AbstractVector)
    return q
end
function blend_projection(proj::EuclideanProjection, set::AbstractAllocationSet,
                          q::AbstractVector, w::AbstractVector)
    return project(proj, set, q, w)
end
public set_allocation_set_constraints!
