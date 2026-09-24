"""
    calibrated(x)
    calibrated(x::AbstractVector)

Whether a penalty slot or a norm-ceiling slot holds a Calibration Rule.

A Calibration Rule resolves against the prior result, so a calibrated slot reads the head's rows. The test reads the slot itself and every slot of the term it holds, which [`calibration_slots`](@ref) lists. A vector is calibrated when one of its entries is.

# Returns

  - `::Bool`: `true` when the slot, or a slot of its term, holds a Calibration Rule.

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

Resolves the slots of a [`ProgrammeAllocationSet`](@ref) that read the head's rows, at every step.

This is the second stage of the set's resolution. [`resolve_allocation_set`](@ref) resolves the slots keyed by name once per fold. This function resolves the rest on the prior result `pr` that is fitted on the rows carrier `rd`, as [`processed_jump_optimiser_attributes`](@ref) and [`assemble_jump_model!`](@ref) resolve them for a batch head. A set that reads no rows has no prior, and its slots come back as they are.

# Algorithm

The method on a prior result runs these steps.

 1. Read `strict` from the current [`ProjectionStep`](@ref).
 2. When `lcse` is keyed by exposure, build `lcsr` from it through [`linear_constraints`](@ref) with the prior's factor regression `pr.rr` and the rows. Otherwise `lcsr` is `lcse`.
 3. Fit the centrality rows `ctr` and the phylogeny result `plr` on the prior and the rows.
 4. Resolve the return floor `ret` against the prior.
 5. Resolve the Calibration Rules of the norm ceilings `l2c`, `linfc` and `lpc`, and of the penalties `l1`, `l2`, `lp` and `linf`, against the prior and the set's solver.
 6. Return the resolved slots and `rd`.

# Arguments

  - `set`: The programme set, resolved once per fold.
  - `pr`: The prior result fitted on `rd`, or `nothing` when the set reads no rows.
  - `rd`: The rows carrier the head holds, a [`ReturnsResult`](@ref).

# Validation

  - A prior result with no carrier. An `IsNothingError` is thrown. The prior is fitted on the carrier, so this pair does not occur in the library.

# Returns

  - `attrs::NamedTuple`: `lcsr`, `ctr`, `plr`, `ret`, `l2c`, `lpc`, `linfc`, `l1`, `l2`, `lp`, `linf`, and the rows carrier `rd`, which is `nothing` when the set reads no rows.

# Related

  - [`resolve_allocation_set`](@ref)
  - [`set_allocation_set_constraints!`](@ref)
  - [`linear_constraints`](@ref)
  - [`centrality_constraints`](@ref)
  - [`phylogeny_constraints`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function resolve_allocation_set_rows(::ProgrammeAllocationSet, ::AbstractPriorResult,
                                     ::Nothing)
    return throw(IsNothingError("a programme set's prior result is fitted on the head's rows carrier, so a result beside no carrier cannot resolve the row-reading slots."))
end
function resolve_allocation_set_rows(set::ProgrammeAllocationSet, ::Nothing, ::Any)
    return (; lcsr = set.lcse, ctr = set.cte, plr = set.ple, ret = set.ret, l2c = set.l2c,
            lpc = set.lpc, linfc = set.linfc, l1 = set.l1, l2 = set.l2, lp = set.lp,
            linf = set.linf, rd = nothing)
end
function resolve_allocation_set_rows(set::ProgrammeAllocationSet, pr::AbstractPriorResult,
                                     rd::ReturnsResult)
    strict = projection_step_strict()
    slv = set.slv
    lcsr = if exposure_keyed(set.lcse)
        linear_constraints(set.lcse, set.sets; datatype = eltype(rd.X), strict = strict,
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
    set_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet, w::AbstractVector, X, pr = nothing)
    set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X)
    set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X, pr)

Adds the constraints of a resolved Allocation Set to the bare projection model.

The model must already hold `w`, `k`, the two scales and the observation count. On both kinds of set the function adds the weight bounds and the budget of one. On a [`ProgrammeAllocationSet`](@ref) it also adds every other kind of the set, through the builders a JuMP optimiser uses and in the order [`assemble_jump_model!`](@ref) runs them. [`assemble_allocation_set!`](@ref) lists that order.

[`projection_programme`](@ref) calls the five-argument form with the prior it fitted and reduced to the Investable Mask. The four-argument form fits the prior on `X` itself. A JuMP head does not call this function. Its Allocation Set Constraint calls [`add_allocation_set_constraints!`](@ref), which runs the same builders under names that do not collide with the head's own. The prior must price every asset the model trades.

# Algorithm

 1. In the four-argument form, fit the set's prior `pr` on `X` with [`allocation_set_prior`](@ref), and call the five-argument form.
 2. Add the weight bounds, the budget of one, and the short and gross budgets with [`set_allocation_set_bounds!`](@ref).
 3. On a programme set, add every other kind with [`assemble_allocation_set!`](@ref) under the empty prefix.

# Arguments

  - $(arg_dict[:model])
  - `set`: The Allocation Set, resolved once per fold.
  - `w`: The Price-Adjusted Allocation the step trades from. It is the reference of the turnover ceilings and of a tracking benchmark that is not fixed.
  - `X`: The rows carrier the head holds through the period, a [`ReturnsResult`](@ref), or `nothing`.
  - `pr`: The prior result that the slots which read the rows are built on, [`allocation_set_prior`](@ref) of the set on `X`, or `nothing` when the set reads no rows.

# Validation

  - A slot that reads the rows, with `X === nothing`. The four-argument form throws an `ArgumentError`, because a projection gets the rows only inside an Online Update.
  - A prior whose Investable Mask leaves out an asset the model trades. [`assert_set_prior_priced`](@ref) throws an `ArgumentError`.

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
                                         ::AbstractVector, ::Any, ::Any = nothing)::Nothing
    set_allocation_set_bounds!(model, set.wb)
    return nothing
end
function set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                         w::AbstractVector,
                                         X::Option{<:ReturnsResult})::Nothing
    return set_allocation_set_constraints!(model, set, w, X, allocation_set_prior(set, X))
end
function set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                         w::AbstractVector, X::Option{<:ReturnsResult},
                                         pr::Option{<:AbstractPriorResult})::Nothing
    set_allocation_set_bounds!(model, set.wb, set.sbgt, set.gbgt)
    assemble_allocation_set!(model, set, w, X, pr, Symbol(""))
    return nothing
end
"""
    add_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet, w::AbstractVector, X)
    add_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X)

Adds the constraints of a resolved Allocation Set to a JuMP head's model during its assembly.

This is the arm of the Allocation Set Constraint that a [`FollowTheLeader`](@ref) rule appends to its held optimiser. The head's builders have already registered their named entries: the weight bounds, the budgets, and the turnover and tracking-error terms under their indices. So this function adds the set's rows under other names than the bare projection model uses. It adds the bounds, the budget of one, and the short and gross budgets as anonymous rows. It adds every other kind through the same builders as [`set_allocation_set_constraints!`](@ref), in the same order. A builder that takes a prefix gets `:aset_`. A builder that takes an index gets the first index the model has not used. The function registers the head's `w` under the prefix, so the entries of a set's risk ceiling never collide with the head's own risk measures.

The set's penalties and the `p · tr(W)` term of its semidefinite phylogeny go into the Objective Penalty, which the head's objective builder adds to the head's objective. The tracking errors read the head's rows carrier `X` and its row count. The set's prior is fitted on that carrier, as the leader's prior is. The leader's model trades only the Investable Mask of its own prior, so [`assert_set_prior_priced`](@ref) refuses a set prior that prices fewer assets than the leader's.

Some builders register their entries under one fixed name per model: a MIP kind, the exact long-short pin, a norm ceiling, a penalty and an integer phylogeny. State such a kind on the set or on the held optimiser, not on both. When it is on both, the second registration fails with an error that names the entry.

# Algorithm

 1. Add the bounds and the budgets as anonymous rows with [`add_allocation_set_bounds!`](@ref).
 2. On a programme set, fit the set's prior on `X` with [`allocation_set_prior`](@ref), and add every other kind with [`assemble_allocation_set!`](@ref) under the prefix `:aset_`.

# Arguments

  - $(arg_dict[:model])
  - `set`: The Allocation Set, resolved once per fold.
  - `w`: The Price-Adjusted Allocation the step trades from.
  - `X`: The rows carrier the head holds through the period, a [`ReturnsResult`](@ref), or `nothing`.

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
                                         w::AbstractVector,
                                         X::Option{<:ReturnsResult})::Nothing
    add_allocation_set_bounds!(model, set.wb, set.sbgt, set.gbgt)
    assemble_allocation_set!(model, set, w, X, allocation_set_prior(set, X), :aset_)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds every kind of a programme set other than the bounds and the budgets, in the order [`assemble_jump_model!`](@ref) runs them.

Both arms call it after they add the bounds. The bare projection model passes the empty prefix `Symbol("")`. A leader's model passes `:aset_`. Then the builders that take an index use the first free index, and the head's `w` is registered under the prefix.

# Algorithm

 1. Check that the prior prices every asset in `X` with [`assert_set_prior_priced`](@ref).
 2. Resolve the slots that read the rows with [`resolve_allocation_set_rows`](@ref). This gives `lcsr`, `ctr`, `plr`, `ret`, `l2c`, `lpc`, `linfc`, `l1`, `l2`, `lp`, `linf` and `rd`.
 3. When `prefix` is not empty, register the model's `w` under it, with the owner `w_owner = Symbol("")`. Then [`weights_prefix`](@ref) keeps the lifted `W`, `variance_flag` and `rc_variance` in the bare namespace, because the weights are the leader's.
 4. Add the linear rows `lcsr`, then the centrality rows `ctr`.
 5. Add the MIP kinds, then the sub-group MIP kinds.
 6. Add the turnover ceilings, each with its reference replaced by `w`, through [`set_allocation_turnover!`](@ref).
 7. When the set has tracking errors, resolve each against `w`, so that a benchmark that is not fixed takes `w`. Cut each benchmark series to the row count of the prior with [`prefix_tracking_benchmark`](@ref), and add the rows.
 8. Add the norm ceilings `l2c`, `lpc` and `linfc`.
 9. Add the penalties `l1`, `l2`, `lp` and `linf` to the Objective Penalty.
10. Add the risk ceilings with [`set_allocation_risk_ceiling!`](@ref).
11. Add the return floor `ret` with [`set_allocation_return_floor!`](@ref).
12. Add the integer phylogeny rows and the semidefinite phylogeny rows of `plr`.
13. Add the custom constraints.

# Arguments

  - $(arg_dict[:model])
  - `set`: The programme set, resolved once per fold.
  - `w`: The Price-Adjusted Allocation the step trades from.
  - `X`: The rows carrier the head holds through the period, a [`ReturnsResult`](@ref), or `nothing`.
  - `pr`: The set's prior result on `X`, or `nothing` when the set reads no rows.
  - `prefix`: `Symbol("")` on the bare projection model, `:aset_` in a leader's model.

# Returns

  - `nothing`.

# Related

  - [`set_allocation_set_constraints!`](@ref)
  - [`add_allocation_set_constraints!`](@ref)
  - [`resolve_allocation_set_rows`](@ref)
"""
function assemble_allocation_set!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                  w::AbstractVector, X::Option{<:ReturnsResult},
                                  pr::Option{<:AbstractPriorResult},
                                  prefix::Symbol)::Nothing
    bare = prefix == Symbol("")
    assert_set_prior_priced(pr, X)
    (; lcsr, ctr, plr, ret, l2c, lpc, linfc, l1, l2, lp, linf, rd) = resolve_allocation_set_rows(set,
                                                                                                 pr,
                                                                                                 X)
    if !bare
        state_build!(() -> get_w(model), model, prefix, :w)
        state_build!(() -> Symbol(""), model, prefix, :w_owner)
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

Cuts a [`ReturnsTracking`](@ref) benchmark of a programme set to its first `T` entries when it is longer.

The head's rows are the first rows of the fold, up to the current period, so a benchmark series stated over the whole fold aligns with them by its first `T` entries. A benchmark shorter than the rows goes to the builder unchanged, and the builder throws a `DimensionMismatch`. A weights benchmark reads the rows themselves and is returned unchanged. A vector of tracking errors is cut entry by entry.

# Arguments

  - `tr`: A tracking error, a vector of them, or any other value.
  - `T`: The row count of the head's rows.

# Returns

  - `tr'`: `tr` with each returns benchmark cut to `T` entries.

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

Adds a programme set's turnover ceilings, whose reference is already the step's Price-Adjusted Allocation.

On the bare projection model the ceilings take the indices one to the number of ceilings. In a leader's model each ceiling takes the first index that the model has not used, so the set's terms do not replace the head's own.

# Arguments

  - $(arg_dict[:model])
  - `tn`: The turnover ceilings, one [`Turnover`](@ref), a vector of them, or `nothing`.
  - `bare`: `true` on the bare projection model, `false` in a leader's model.

# Returns

  - `nothing`.

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

Adds a programme set's return floor.

For each term, the per-index arm of [`set_return_constraints!`](@ref) writes the return expression on the prior's expected returns and the row of its `settings.lb`. The term's scale is set to one first. On the bare projection model the terms take the indices one to the number of terms. In a leader's model each term takes the first index the model has not used. The function does not use a term in the objective, as a scalarised return or the normalisation of a ratio, because the objective of the projection is the geometry's divergence.

# Arguments

  - $(arg_dict[:model])
  - `ret`: The return floor, one return estimator, a vector of them, or `nothing`.
  - `pr`: The set's prior result.
  - `bare`: `true` on the bare projection model, `false` in a leader's model.
  - `kwargs...`: Forwarded to [`set_return_constraints!`](@ref). The rows carrier arrives as `rd`.

# Returns

  - `nothing`.

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

Adds an Allocation Set's weight bounds, its budget of one, and its short and gross budgets to a model that already names its own.

The rows are the inequalities that [`set_allocation_set_bounds!`](@ref) writes, registered under no name. The short and gross budgets read the model's long and short parts when the model holds them. Otherwise the function makes two anonymous vectors of parts and bounds them by `w`.

# JuMP formulation

## Variables

  - `w`, `k`: read from the model.
  - `lw`, `sw`: long and short parts. They are read from the model when it holds them. Otherwise, when the set has a short or a gross budget, they are two anonymous vectors of ``N`` variables, each bounded below by zero.

## Constraints

Every row is registered under no name.

  - ``s_c (\\boldsymbol{w} - k \\boldsymbol{l}) \\geq 0``, when an entry of ``\\boldsymbol{l}`` is finite.
  - ``s_c (\\boldsymbol{w} - k \\boldsymbol{u}) \\leq 0``, when an entry of ``\\boldsymbol{u}`` is finite.
  - ``s_c \\left(\\sum_{i=1}^{N} w_i - k\\right) = 0``.
  - ``s_c (\\boldsymbol{w} - \\boldsymbol{lw}) \\leq 0`` and ``s_c (\\boldsymbol{w} + \\boldsymbol{sw}) \\geq 0``, when the function makes the parts.
  - The rows of [`anonymous_budget_rows!`](@ref) on ``\\sum_i sw_i`` for the short budget, and on ``\\sum_i (lw_i + sw_i)`` for the gross budget.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])
  - $(math_dict[:w_port])
  - $(math_dict[:N])
  - ``\\boldsymbol{l}``, ``\\boldsymbol{u}``: Lower and upper weight bounds of the set, resolved.
  - ``\\boldsymbol{lw}``, ``\\boldsymbol{sw}``: Long and short parts of ``\\boldsymbol{w}``, each ``N \\times 1`` and non-negative.

# Arguments

  - $(arg_dict[:model])
  - `wb`: The set's resolved [`WeightBounds`](@ref).
  - `sbgt`: The short budget, a number, a [`BudgetRange`](@ref), or `nothing`.
  - `gbgt`: The gross budget, a number, a [`BudgetRange`](@ref), or `nothing`.

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

Adds one budget on an expression as rows under no name.

A number fixes the expression at `k · bgt`. A [`BudgetRange`](@ref) bounds it on each side that is set. `nothing` adds no row.

# JuMP formulation

## Variables

  - `k`: read from the model.

## Constraints

Every row is registered under no name.

  - ``s_c (e - k b) = 0``, when the budget is a number.
  - ``s_c (e - k b_l) \\geq 0`` and ``s_c (e - k b_u) \\leq 0``, when the budget is a range, each row only when its bound is set.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])
  - ``e``: The expression the budget bounds.
  - ``b``: The budget, when it is a number.
  - ``b_l``, ``b_u``: Lower and upper bounds of the range.

# Arguments

  - $(arg_dict[:model])
  - `expr`: The expression the budget bounds.
  - `bgt`: A number, a [`BudgetRange`](@ref), or `nothing`.

# Returns

  - `nothing`.

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

Returns the first index `i`, counted from one, at which the Model State holds no entry `name`.

A builder keyed by an index uses it to add the set's term next to the head's own terms, which hold the indices from one up.

# Returns

  - `i::Int`: The first free index.

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

Adds an Allocation Set's weight bounds, its budget of one, and its short and gross budgets to the bare projection model.

[`set_weight_constraints!`](@ref) writes the bounds, the short budget and the gross budget, with no net budget. A negative lower bound then builds the long and short parts of `w` and leaves the long part free. Under [`JuMPOptimiser`](@ref), a net budget with no short budget fixes the long part at the net budget, which forbids every short position. An Allocation Set lets the book hold short positions, so the function passes no net budget there. It adds `Σw = k` after that, through [`set_budget_constraints!`](@ref).

# Algorithm

 1. Check the bounds against the budget with [`assert_feasible_bounds`](@ref).
 2. Add the bounds, the short budget and the gross budget with [`set_weight_constraints!`](@ref), with no net budget.
 3. Add the budget `Σw = k` with [`set_budget_constraints!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `wb`: The set's resolved [`WeightBounds`](@ref).
  - `sbgt`: The short budget, a number, a [`BudgetRange`](@ref), or `nothing`.
  - `gbgt`: The gross budget, a number, a [`BudgetRange`](@ref), or `nothing`.

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
    set_projection_objective!(model::JuMP.Model, proj::DiagonalProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::TsallisProjection, q::AbstractVector)
    set_projection_objective!(model::JuMP.Model, proj::LogBarrierProjection, q::AbstractVector)

Sets the objective of the projection programme to the geometry's divergence from the raw step `q`, plus the Objective Penalty.

Each arm writes its divergence ``f`` as a conic model and hands it to [`set_divergence_objective!`](@ref), which adds the penalty and sets the objective.

# Mathematical definition

```math
\\begin{align}
f_{\\mathrm{E}}(\\boldsymbol{w}) &= \\lVert \\boldsymbol{w} - \\boldsymbol{q} \\rVert_2\\,, \\\\
f_{\\mathrm{E}}^{\\pi}(\\boldsymbol{w}) &= \\frac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{q} \\rVert_2^2\\,, \\\\
f_{\\mathrm{G}}(\\boldsymbol{w}) &= \\lVert \\mathbf{G} (\\boldsymbol{w} - \\boldsymbol{q}) \\rVert_2\\,, \\\\
f_{\\mathrm{G}}^{\\pi}(\\boldsymbol{w}) &= \\frac{1}{2} \\lVert \\mathbf{G} (\\boldsymbol{w} - \\boldsymbol{q}) \\rVert_2^2\\,, \\\\
f_{\\mathrm{KL}}(\\boldsymbol{w}) &= \\sum_{i \\in P} w_i \\log \\frac{w_i}{q_i}\\,, \\\\
f_{\\alpha}(\\boldsymbol{w}) &= \\frac{1}{1 - \\alpha} \\sum_{i \\in P} \\left( q_i^{\\alpha - 1} w_i - \\frac{w_i^{\\alpha}}{\\alpha} \\right)\\,, \\\\
f_{\\mathrm{IS}}(\\boldsymbol{w}) &= \\sum_{i \\in P} \\left( \\frac{w_i}{q_i} - \\log w_i \\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:q_raw])
  - ``\\pi``: Objective Penalty, the sum of the set's penalties and of the other terms added to the objective before it.
  - ``f_{\\mathrm{E}}``, ``f_{\\mathrm{E}}^{\\pi}``: Divergence of [`EuclideanProjection`](@ref), without and with an Objective Penalty.
  - ``f_{\\mathrm{G}}``, ``f_{\\mathrm{G}}^{\\pi}``: Divergence of [`GramProjection`](@ref) and of [`DiagonalProjection`](@ref), without and with an Objective Penalty.
  - ``\\mathbf{G}``: Upper Cholesky factor of the geometry's Gram matrix ``\\mathbf{A}``, so ``\\mathbf{G}^\\intercal \\mathbf{G} = \\mathbf{A}``. Under [`DiagonalProjection`](@ref) it is ``\\mathrm{diag}(\\sqrt{\\boldsymbol{h}})`` and ``\\mathbf{A} = \\mathrm{diag}(\\boldsymbol{h})``.
  - ``f_{\\mathrm{KL}}``: Relative entropy, the divergence of [`EntropicProjection`](@ref).
  - ``f_{\\alpha}``: Tsallis divergence, the divergence of [`TsallisProjection`](@ref).
  - ``\\alpha``: Power of the Tsallis potential, in ``(0, 1)``.
  - ``f_{\\mathrm{IS}}``: Itakura–Saito divergence, the divergence of [`LogBarrierProjection`](@ref).
  - ``P = \\{i : q_i > 0\\}``: Positive entries of the raw step. Under the entropic, Tsallis and log-barrier geometries every ``w_i`` with ``i \\notin P`` is zero.
  - $(math_dict[:Psi_pot])
  - $(math_dict[:D_Psi_breg])

Each ``f`` differs from the Bregman divergence ``D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q})`` of its geometry's potential by a term that is constant in ``\\boldsymbol{w}`` on the budget ``\\sum_i w_i = 1``, so each has the minimiser of ``D_\\Psi`` and a penalty added to it weighs against ``D_\\Psi`` itself. ``f_{\\mathrm{E}}^{\\pi}`` and ``f_{\\mathrm{G}}^{\\pi}`` are these Bregman divergences, of ``\\tfrac{1}{2} \\lVert \\boldsymbol{w} \\rVert_2^2`` and ``\\tfrac{1}{2} \\boldsymbol{w}^\\intercal \\mathbf{A} \\boldsymbol{w}``. The norm ``f_{\\mathrm{E}}`` has the same minimiser as ``f_{\\mathrm{E}}^{\\pi}`` when no penalty is present, and ``f_{\\mathrm{G}}`` has the same minimiser as ``f_{\\mathrm{G}}^{\\pi}``. A penalty ``\\pi`` added to the norm does not move the minimiser off a feasible ``\\boldsymbol{q}`` while the dual norm of ``\\nabla \\pi(\\boldsymbol{q})`` is at most one, because the subdifferential of a norm at zero is its dual unit ball. Added to the squared norm, it gives the proximal step.

# JuMP formulation

## Variables

  - `w`: read from the model.
  - `t_proj`: created. It is a scalar under the Euclidean, Gram, diagonal and entropic arms, and a vector with one entry for each index in ``P`` under the Tsallis and log-barrier arms.

## Constraints

  - Euclidean, Gram and diagonal arms: the row of [`set_distance_cone!`](@ref) on ``\\boldsymbol{x} = s_c (\\boldsymbol{w} - \\boldsymbol{q})`` or ``\\boldsymbol{x} = s_c \\mathbf{G} (\\boldsymbol{w} - \\boldsymbol{q})``.
  - Entropic arm, `proj_zero`: ``s_c w_i = 0`` for each ``i \\notin P``, when such an ``i`` exists.
  - Entropic arm, `proj_rec`: ``s_c t \\geq \\sum_{i \\in P} s_c w_i \\log (w_i / q_i)``, the relative entropy cone over ``(s_c t, s_c \\boldsymbol{q}_P, s_c \\boldsymbol{w}_P)``.
  - Tsallis arm, under no name: ``t_j \\leq w_j^{\\alpha}`` for each ``j \\in P``, the power cone of power ``\\alpha`` over ``(s_c w_j, s_c, s_c t_j)``.
  - Log-barrier arm, under no name: ``t_j \\leq \\log w_j`` for each ``j \\in P``, the exponential cone over ``(s_c t_j, s_c, s_c w_j)``.
  - Tsallis and log-barrier arms: the rows of [`barrier_objective_entries`](@ref), which fix ``w_i = 0`` for each ``i \\notin P``.

The divergence each arm hands to [`set_divergence_objective!`](@ref) is ``t`` under the Euclidean, Gram, diagonal and entropic arms, ``\\frac{1}{1 - \\alpha} \\left( \\sum_{j \\in P} q_j^{\\alpha - 1} w_j - \\frac{1}{\\alpha} \\sum_{j \\in P} t_j \\right)`` under the Tsallis arm, and ``\\sum_{j \\in P} w_j / q_j - \\sum_{j \\in P} t_j`` under the log-barrier arm.

Where:

  - $(math_dict[:sc_scale])
  - ``t``, ``t_j``: The entries of `t_proj`.
  - ``\\boldsymbol{q}_P``, ``\\boldsymbol{w}_P``: The entries of ``\\boldsymbol{q}`` and ``\\boldsymbol{w}`` at the indices in ``P``.

# Arguments

  - $(arg_dict[:model])
  - `proj`: The Projection Geometry.
  - `q`: The raw step.

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
    # The bracket is `(1 − α)` times the divergence up to a constant. The division restores
    # the divergence, so a penalty weighs against it as under every other geometry.
    set_divergence_objective!(model,
                              (LinearAlgebra.dot(q[pos] .^ (a - 1), w[pos]) -
                               sum(t_proj) / a) / (1 - a), so)
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

Bounds a projection's distance variable `t` below by the norm of the scaled deviation `x`, or by half its square when the model holds an Objective Penalty.

Without a penalty the function writes the second-order cone. Its minimum is sharp, and the solver finds it to a tight tolerance. With a penalty it writes the rotated second-order cone, because a penalty added to the norm does not move the minimiser off a feasible raw step while its gradient is small, and added to half the square it gives the proximal step. The penalty is present when the set's penalties, a custom objective term or the `p · tr(W)` term of a semidefinite phylogeny were added before the objective. [`set_projection_objective!`](@ref) states the two divergences.

# JuMP formulation

## Variables

  - `t`: read. It is the caller's distance variable.

## Constraints

  - `proj_soc`: ``(s_c t, \\boldsymbol{x}) \\in \\mathcal{K}_{\\mathrm{SOC}}``, that is ``t \\geq \\lVert \\boldsymbol{x} \\rVert_2 / s_c``, when the model holds no Objective Penalty.
  - `proj_rsoc`: ``(s_c t, s_c, \\boldsymbol{x}) \\in \\mathcal{K}_{\\mathrm{RSOC}}``, that is ``t \\geq \\lVert \\boldsymbol{x} \\rVert_2^2 / (2 s_c^2)``, when it holds one.

Where:

  - $(math_dict[:sc_scale])
  - ``\\boldsymbol{x}``: The scaled deviation, ``s_c (\\boldsymbol{w} - \\boldsymbol{q})`` or ``s_c \\mathbf{G} (\\boldsymbol{w} - \\boldsymbol{q})``, so the two rows give ``t \\geq \\lVert \\boldsymbol{w} - \\boldsymbol{q} \\rVert_2`` and ``t \\geq \\tfrac{1}{2} \\lVert \\boldsymbol{w} - \\boldsymbol{q} \\rVert_2^2`` in the Euclidean geometry.
  - ``\\mathcal{K}_{\\mathrm{SOC}} = \\{(u, \\boldsymbol{v}) : u \\geq \\lVert \\boldsymbol{v} \\rVert_2\\}``: Second-order cone.
  - ``\\mathcal{K}_{\\mathrm{RSOC}} = \\{(u, v, \\boldsymbol{z}) : 2 u v \\geq \\lVert \\boldsymbol{z} \\rVert_2^2,\\, u \\geq 0,\\, v \\geq 0\\}``: Rotated second-order cone.

# Arguments

  - $(arg_dict[:model])
  - `t`: The distance variable.
  - `x`: The scaled deviation, a vector of affine expressions.
  - `sc`: The constraint scale.

# Returns

  - `nothing`.

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

Sets the projection programme's objective to the minimum of the geometry's divergence `div` plus the Objective Penalty, times the objective scale.

The penalty holds the set's penalties, its custom objective terms and the `p · tr(W)` term of its semidefinite phylogeny. It enters through [`add_penalty_to_objective!`](@ref), which every JuMP head's objective also uses. With no penalty the objective is the divergence alone, which is the projection that the papers of the rules state.

# JuMP formulation

## Expressions

  - `obj_expr`: ``d + \\pi``. The function registers ``d`` first and then adds the model's `op` when it holds one.

## Objective

  - `Min`: ``s_o`` times `obj_expr`.

Where:

  - ``d``: The divergence `div` of the geometry.
  - ``\\pi``: The Objective Penalty `op`, zero when the model holds none.
  - $(math_dict[:so_scale])

# Arguments

  - $(arg_dict[:model])
  - `div`: The divergence, an affine expression or a variable.
  - `so`: The objective scale.

# Returns

  - `nothing`.

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

Prepares the entries that a barrier objective is written over.

It refuses a raw step with a negative entry or with no positive entry. It fixes the weight of every zero entry of `q` at zero. It returns the positive indices, the model's `w` and the two scales.

# JuMP formulation

## Variables

  - `w`: read from the model.

## Constraints

  - Under no name: ``s_c w_i = 0`` for each ``i`` with ``q_i = 0``.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:q_raw])

# Arguments

  - $(arg_dict[:model])
  - `q`: The raw step.

# Validation

  - `all(>= 0, q)`. A `DomainError` is thrown otherwise.
  - `sum(q) > 0`. A `DomainError` is thrown otherwise.

# Returns

  - `(pos, w, sc, so)::Tuple`: The indices of the positive entries of `q`, the model's `w`, the constraint scale and the objective scale.

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

Returns the solver a projection programme runs on.

A [`GramProjection`](@ref) carries its own solver, and the programme uses it on every set. Every other geometry uses the programme set's solver. On a [`BoundedAllocationSet`](@ref) every geometry other than the Gram one is a scalar root with no programme, so the function throws an `ArgumentError` that names the geometry. [`project`](@ref) does not call it in that case.

# Returns

  - `slv`: The solver, or a vector of solvers.

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

Tells whether the set's constraints can be built on the rows `X` of the step.

A programme set that fits its prior on the rows cannot be built while the head holds fewer than two rows, because a covariance of one observation does not exist. Every other set can always be built.

# Returns

  - `reason`: `nothing` when the constraints can be built, and otherwise the reason for the Held Step as a `String`.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`projection_programme`](@ref)
  - [`HeldStep`](@ref)
"""
function allocation_set_ready(::AbstractAllocationSet, ::Any)
    return nothing
end
function allocation_set_ready(set::ProgrammeAllocationSet, X::Option{<:ReturnsResult})
    if isnothing(rows_needed(set)) && !isnothing(X) && size(X.X, 1) < 2
        return "the set's prior estimator fits on the head's rows and a covariance of one observation does not exist, so the step trades nothing until the head holds two"
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Projects the raw step `q` onto the Allocation Set in the geometry's divergence, as a JuMP programme.

A solved programme returns its weights. A programme that does not solve, or whose constraints cannot be built on the step's rows ([`allocation_set_ready`](@ref)), is a Held Step. Then [`record_held_step!`](@ref) writes the record to the current [`ProjectionStep`](@ref), and the function returns a copy of `w`, the book the fund already holds.

A set that reads the rows fits its prior on the step's rows carrier, as a batch head does. The programme then runs on the prior's Investable Mask, also as a batch head does. It solves over the assets the prior prices, and gives a zero weight to every other asset. Under a time-varying panel, an asset without a listing for part of the window is outside the universe of a plain prior until the window no longer holds that part. So the programme gives it a zero, which is the one zero in the recursion's allocation that the rule's step did not make. A batch head gives the same zero. The asset comes back at the first step at which the prior prices it, and the rule's geometry decides whether the raw step then gives it weight, as for a zero in the Start Allocation. A set that reads no rows fits no prior and solves over the full universe.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}^{+} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q}) + \\pi(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}^{+}``: The projected allocation.
  - $(math_dict[:W_aset]) Under an Investable Mask it also holds ``w_i = 0`` for every asset ``i`` outside the mask.
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:Psi_pot])
  - $(math_dict[:q_raw])
  - ``\\pi``: Objective Penalty of the set, zero when the set has no penalty.

[`set_projection_objective!`](@ref) states the objective each geometry writes for ``D_\\Psi``.

# Algorithm

 1. Read the rows `X` of the current [`ProjectionStep`](@ref).
 2. When [`allocation_set_ready`](@ref) gives a reason, record the Held Step with it and return a copy of `w`. Stop.
 3. Fit the set's prior `pr` on `X` with [`allocation_set_prior`](@ref), and read its Investable Mask `imsk`.
 4. Reduce `set`, `X` and `pr` to the mask with [`programme_investable_reduction`](@ref). View `q` and `w` at the mask, which gives `q` and `wr`.
 5. Make a bare model with the set's two scales and the row count of `X`, zero when there are no rows. Register `k = 1`, and the vector `w` of weight variables.
 6. Add the set's constraints with [`set_allocation_set_constraints!`](@ref), with `wr` as the reference.
 7. Add the set's custom objective terms with [`add_custom_objective_term!`](@ref), with the geometry as the objective and the set as the owner.
 8. Set the objective with [`set_projection_objective!`](@ref).
 9. Solve on [`projection_solver`](@ref). When the solve succeeds, return the solution with a zero at every asset outside the mask. Otherwise record the Held Step with the solver trials, and return a copy of `w`.

# JuMP formulation

## Variables

  - `w`: created, with the base name `w`, one variable for each asset in the mask. It is registered in the Model State under the key `w`.

## Expressions

  - `k`: the constant one.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])

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
    pr = allocation_set_prior(set, X)
    imsk = prior_investable_mask(pr)
    set, X, pr = programme_investable_reduction(imsk, set, X, pr)
    q = investable_weights_view(imsk, q)
    wr = investable_weights_view(imsk, w)
    model = JuMP.Model()
    set_model_scales!(model, projection_scale(set, :sc), projection_scale(set, :so))
    set_model_observations!(model, isnothing(X) ? 0 : size(X.X, 1))
    JuMP.@expression(model, k, 1)
    wv = state_set!(model, Symbol(""), :w,
                    JuMP.@variable(model, [1:length(q)], base_name = "w"))
    set_allocation_set_constraints!(model, set, wr, X, pr)
    add_custom_objective_term!(model, proj, allocation_set_cobj(set), set, nothing)
    set_projection_objective!(model, proj, q)
    res = optimise_JuMP_model!(model, projection_solver(proj, set))
    if res.success
        return expand_investable_weights(imsk, JuMP.value.(wv))
    end
    record_held_step!("the projection onto the `$(nameof(typeof(set)))` in the `$(nameof(typeof(proj)))` geometry did not solve, and the step trades nothing",
                      res.trials)
    return copy(w)
end
"""
    programme_investable_reduction(imsk::Nothing, set::AbstractAllocationSet, X::Option{<:ReturnsResult}, pr)
    programme_investable_reduction(imsk::BitVector, set::AbstractAllocationSet, X::ReturnsResult, pr::AbstractPriorResult)
    programme_investable_reduction(imsk::BitVector, set::AbstractAllocationSet, X, pr)

Reduces the set, the rows carrier and the set's prior to the prior's Investable Mask.

The projection programme then runs on the assets the prior can price, as a batch head does with [`investable_reduction`](@ref). The caller views the raw step and the Price-Adjusted Allocation at the same mask with [`investable_weights_view`](@ref).

Dispatch selects one of three methods. A `nothing` mask means that every asset is investable, which is true at every step on a static panel and at every step of a set that reads no rows. That method returns its arguments unchanged. The `BitVector` method takes the views at `findall(imsk)`. The third method refuses a mask with no carrier or no prior, because a mask comes from a prior fitted on a carrier, so this pair does not occur in the library. The set is viewed against the carrier's full returns matrix, because a view of a tracking estimator needs it. The Price-Adjusted Allocation is cut to the mask and is not renormalised, as the batch reduction cuts a turnover reference. So an asset the prior cannot price leaves the reference, and the programme allocates the budget of one over the investable assets. The function writes no warning for such an asset, because the Investable Mask of the read-out already shows it at every step.

# Validation

  - A mask with no carrier or no prior. An `IsNothingError` is thrown.

# Returns

  - `(set, X, pr)::Tuple`: The three reduced to the mask, or unchanged.

# Related

  - [`projection_programme`](@ref)
  - [`allocation_set_prior`](@ref)
  - [`prior_investable_mask`](@ref)
  - [`investable_weights_view`](@ref)
  - [`expand_investable_weights`](@ref)
"""
function programme_investable_reduction(::Nothing, set::AbstractAllocationSet,
                                        X::Option{<:ReturnsResult}, pr)
    return set, X, pr
end
function programme_investable_reduction(imsk::BitVector, set::AbstractAllocationSet,
                                        X::ReturnsResult, pr::AbstractPriorResult)
    idx = findall(imsk)
    return port_opt_view(set, idx, X.X), port_opt_view(X, idx), port_opt_view(pr, idx)
end
function programme_investable_reduction(::BitVector, ::AbstractAllocationSet, ::Any, ::Any)
    return throw(IsNothingError("an Investable Mask is derived from a prior fitted on the head's rows carrier, so a mask beside no carrier or no prior result cannot be reduced on."))
end
"""
    projection_scale(set::ProgrammeAllocationSet, f::Symbol)
    projection_scale(set::BoundedAllocationSet, f::Symbol)

Returns the constraint scale `:sc` or the objective scale `:so` that a projection programme registers on its model.

A programme set gives its own field. A bounded set carries no scale, so it gives one.

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

Returns the custom objective terms that a projection programme adds to its objective.

A programme set gives its `cobj`. A bounded set carries no objective term, so it gives `nothing`.

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

Projects a raw step onto the Allocation Set by the programme of [`projection_programme`](@ref).

These are the programme arms of the Constrained Update. They cover every pair of a geometry and a set except the scalar roots on the bounded set. Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref), [`clip_at_zero`](@ref) sets every negative entry of the answer to zero. The solver can give an entry slightly below zero, within its tolerance, and the next step must not read it.

# Validation

  - Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref): `all(>= 0, lb)` over the resolved bounds. A `DomainError` is thrown otherwise.

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

Projects the blend of an [`ExpertMixture`](@ref) onto the head's set a second time, when the set needs it.

On a [`BoundedAllocationSet`](@ref) dispatch returns the blend unchanged. The bounds and the budget define a convex set, so a convex blend of allocations in it is in it too, and the projection is the identity. On every other set the function calls [`project`](@ref), which gives the repair that a turnover ceiling or a MIP kind needs.

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
