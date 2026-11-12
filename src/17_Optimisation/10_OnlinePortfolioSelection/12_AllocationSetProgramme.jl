"""
    calibrated(x)
    calibrated(x::AbstractVector)

Whether a penalty slot or a norm-ceiling slot holds a Calibration Rule.

A Calibration Rule resolves against the prior result, so a calibrated slot reads the head's rows. The test reads the slot itself and every slot of the term it holds, which [`calibration_slots`](@ref) lists. A vector is calibrated when one of its entries is.

# Returns

  - `::Bool`: `true` when the slot, or a slot of its term, holds a Calibration Rule.

# Related

  - [`rows_needed`](@ref)
  - [`resolve_constraint_block`](@ref)
  - [`is_calibration_rule`](@ref)
"""
function calibrated(x::Any)
    return is_calibration_rule(x) || any(is_calibration_rule, values(calibration_slots(x)))
end
function calibrated(x::AbstractVector)
    return any(calibrated, x)
end
"""
    set_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet, w::AbstractVector, X, pr = nothing)
    set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X)
    set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X, pr)

Adds the constraints of a resolved Allocation Set to the bare projection model.

The model must already hold `w`, `k`, the two scales and the observation count. On both kinds of set the function adds the weight bounds and the budget of one. On a [`ProgrammeAllocationSet`](@ref) the projection is the owner of the model, and the set is the owner of its block. The projection writes the budget of one, and the block builder [`set_block_constraints!`](@ref) adds every kind of the block, which a JuMP optimiser adds through the same builder. The bounds builder [`set_allocation_risk_return_bounds!`](@ref) adds the risk ceilings and the return floor of the set between the two phases of the block.

[`projection_programme`](@ref) calls the five-argument form with the prior it fitted and reduced to the Investable Mask. The four-argument form fits the prior on `X` itself. A JuMP head does not call this function. Its Allocation Set Constraint calls [`add_allocation_set_constraints!`](@ref), which runs the same builders under the prefix `:aset_`. The prior must price every asset the model trades.

# Algorithm

 1. In the four-argument form, fit the set's prior `pr` on `X` with [`allocation_set_prior`](@ref), and call the five-argument form.
 2. On a bounded set, add the weight bounds and the budget of one with [`set_allocation_set_bounds!`](@ref), and return `set.wb` with no block Result.
 3. On a programme set, resolve the step with [`resolve_allocation_set_step`](@ref), which gives the set after `factory(set, w)` and the Result `cbr` of its block. Add the budget of one with [`set_budget_constraints!`](@ref). Add the block, the ceilings and the floor with [`add_allocation_set_block!`](@ref) under the empty prefix. Return the bounds of the step and `cbr`.

# Arguments

  - $(arg_dict[:model])
  - `set`: The Allocation Set, resolved once per fold.
  - `w`: The Price-Adjusted Allocation the step trades from. It is the reference of the turnover ceilings and of a tracking benchmark that is not fixed.
  - `X`: The rows the head holds through the period, a [`ReturnsResult`](@ref), or `nothing`.
  - `pr`: The prior result that the slots which read the rows are built on, [`allocation_set_prior`](@ref) of the set on `X`, or `nothing` when the set reads no rows.

# Validation

  - A slot that reads the rows, with `X === nothing`. The four-argument form throws an `ArgumentError`, because a projection gets the rows only inside an Online Update.
  - A prior whose Investable Mask leaves out an asset the model trades. [`assert_set_prior_priced`](@ref) throws an `ArgumentError`.

# Returns

  - `(wb, cbr)::Tuple`: The weight bounds of the step that the model holds, and the [`ResolvedConstraintBlock`](@ref) of the set, or `nothing` for a bounded set. [`projection_programme`](@ref) reads the floor of `wb` under a barrier geometry, and gives `cbr` to the custom objective terms of the set.

**Breaking change.** This function returned `nothing` before a programme set took a bound that reads the rows, and then the bounds of the step alone. It returns the bounds and the block Result, because a custom objective term of the set reads the Result of its block. A programme set now gives its `ccnt` and its `cobj` the Price-Adjusted Allocation of the step through `factory(set, w)`, as it gave its turnover and its tracking errors before.

# Related

  - [`ProgrammeAllocationSet`](@ref)
  - [`BoundedAllocationSet`](@ref)
  - [`add_allocation_set_constraints!`](@ref)
  - [`assemble_jump_model!`](@ref)
  - [`set_allocation_risk_ceiling!`](@ref)
"""
function set_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet,
                                         ::AbstractVector, ::Any, ::Any = nothing)::Tuple
    set_allocation_set_bounds!(model, set.wb)
    return set.wb, nothing
end
function set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                         w::AbstractVector,
                                         X::Option{<:ReturnsResult})::Tuple
    return set_allocation_set_constraints!(model, set, w, X, allocation_set_prior(set, X))
end
function set_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                         w::AbstractVector, X::Option{<:ReturnsResult},
                                         pr::Option{<:AbstractPriorResult})::Tuple
    setw, cbr = resolve_allocation_set_step(set, w, X, pr)
    # The bare projection owns the budget of one, so it writes the row itself and gives the
    # block no net budget (ADR 0237).
    set_budget_constraints!(model, 1, get_w(model))
    add_allocation_set_block!(model, setw, cbr, Symbol(""))
    return cbr.wb, cbr
end
"""
    add_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet, w::AbstractVector, X)
    add_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet, w::AbstractVector, X)

Adds the constraints of a resolved Allocation Set to a JuMP head's model during its assembly.

This is the arm of the Allocation Set Constraint that a [`FollowTheLeader`](@ref) rule appends to its held optimiser. The custom constraints run in phase 2 of the block of the leader, after [`mark_risk_minimised!`](@ref), so the set's block, its ceilings and its floor all come after the risk step of the leader. The block of the set takes the prefix `:aset_`: the block builder [`set_block_constraints!`](@ref) registers the head's `w` under it and adds every row of the set there, the bounds and the short and gross budgets among them. A kind on both the set and the held optimiser gives two sets of rows and no name collision. The MIP bits, the long and short parts and the lifted matrix belong to the weights, so the set reads the leader's.

The set's penalties and the `p · tr(W)` term of its semidefinite phylogeny go into the Objective Penalty, which the head's objective builder adds to the head's objective. The tracking errors read the rows `X` that the head holds, and their row count. The set's prior is fitted on those rows, as the leader's prior is. The leader's model trades only the Investable Mask of its own prior, so [`assert_set_prior_priced`](@ref) refuses a set prior that prices fewer assets than the leader's.

The online head owns the budget of one, so this function writes `Σw = k` as the row `aset_bgt`. Every row of the set is registered in the model under `:aset_`, so a reader of the model can find it.

# Algorithm

 1. On a bounded set, add the bounds and the budget under `:aset_` with [`add_allocation_set_bounds!`](@ref).
 2. On a programme set, fit the set's prior on `X` with [`allocation_set_prior`](@ref), and resolve the step with [`resolve_allocation_set_step`](@ref). Add `Σw = k` under `:aset_` with [`set_budget_constraints!`](@ref). Add the block, the ceilings and the floor with [`add_allocation_set_block!`](@ref) under the prefix `:aset_`.

# Arguments

  - $(arg_dict[:model])
  - `set`: The Allocation Set, resolved once per fold.
  - `w`: The Price-Adjusted Allocation the step trades from.
  - `X`: The rows the head holds through the period, a [`ReturnsResult`](@ref), or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`set_allocation_set_constraints!`](@ref)
  - [`AllocationSetConstraint`](@ref)
  - [`FollowTheLeader`](@ref)
"""
function add_allocation_set_constraints!(model::JuMP.Model, set::BoundedAllocationSet,
                                         ::AbstractVector, ::Any)::Nothing
    add_allocation_set_bounds!(model, set.wb)
    return nothing
end
function add_allocation_set_constraints!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                         w::AbstractVector,
                                         X::Option{<:ReturnsResult})::Nothing
    setw, cbr = resolve_allocation_set_step(set, w, X, allocation_set_prior(set, X))
    # The online head owns the budget of one (ADR 0159). Until the leader re-solve passes the
    # set as a block of the entry (#2047), the set writes the row in the leader's model.
    set_budget_constraints!(model, 1, get_w(model); prefix = :aset_)
    add_allocation_set_block!(model, setw, cbr, :aset_)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolves a programme set for one step of an online head, to the set after `factory(set, w)` and the [`ResolvedConstraintBlock`](@ref) of its block.

Both arms of the set call it: the bare projection, [`set_allocation_set_constraints!`](@ref), and the model of a leader, [`add_allocation_set_constraints!`](@ref). Then [`add_allocation_set_block!`](@ref) adds the Result with the block builder.

# Algorithm

 1. Check that the prior prices every asset in `X` with [`assert_set_prior_priced`](@ref).
 2. Give the set the Price-Adjusted Allocation `w` with `factory(set, w)`, so a turnover ceiling, a tracking error, a risk ceiling, a custom constraint and a custom objective term that read the previous weights read the book of the step.
 3. Resolve stage 2 of the block of the set against `pr` and `X` with [`resolve_constraint_block`](@ref), under the solver of the set and the `strict` flag of the current [`ProjectionStep`](@ref).
 4. Check the bounds of the step against the budget of one with [`assert_feasible_bounds`](@ref). A bound that reads the rows resolves only here ([`bounds_read_rows`](@ref)).

# Arguments

  - `set`: The programme set, resolved once per fold.
  - `w`: The Price-Adjusted Allocation the step trades from.
  - `X`: The rows the head holds through the period, a [`ReturnsResult`](@ref), or `nothing`.
  - `pr`: The set's prior result on `X`, or `nothing` when the set reads no rows.

# Validation

  - The rules of [`assert_set_prior_priced`](@ref), [`resolve_constraint_block`](@ref) and [`assert_feasible_bounds`](@ref).

# Returns

  - `(set', cbr)::Tuple`: The set after `factory(set, w)`, and the Result of its block.

# Related

  - [`add_allocation_set_block!`](@ref)
  - [`resolve_constraint_block`](@ref)
"""
function resolve_allocation_set_step(set::ProgrammeAllocationSet, w::AbstractVector,
                                     X::Option{<:ReturnsResult},
                                     pr::Option{<:AbstractPriorResult})::Tuple
    assert_set_prior_priced(pr, X)
    setw = factory(set, w)
    cbr = resolve_constraint_block(setw.cbe, pr, X; slv = setw.slv,
                                   strict = projection_step_strict())
    assert_feasible_bounds(cbr.wb)
    return setw, cbr
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds the block of a programme set to a model with the block builder, and the set's risk ceilings and return floor with its bounds builder.

The set is the owner of its block, so the builder gets no net budget and no fees, as the set had none before the block. The owner of the model writes the budget of one. The bare projection passes the empty prefix `Symbol("")`, and a leader's model passes `:aset_`, so the rows of the set sit beside the rows of the leader with no name collision.

# Algorithm

 1. Add phase 1 of the block with [`set_block_constraints!`](@ref), with the set as the owner and its tracking errors from [`allocation_set_tracking`](@ref).
 2. Add the risk ceilings and the return floor with [`set_allocation_risk_return_bounds!`](@ref).
 3. Add phase 2 of the block with [`set_block_post_risk_constraints!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `set`: The programme set, after `factory(set, w)`.
  - `cbr`: The Result of its block, from [`resolve_allocation_set_step`](@ref).
  - `prefix`: `Symbol("")` on the bare projection model, `:aset_` in a leader's model.

# Returns

  - `nothing`.

# Related

  - [`resolve_allocation_set_step`](@ref)
  - [`set_block_constraints!`](@ref)
  - [`set_allocation_risk_return_bounds!`](@ref)
"""
function add_allocation_set_block!(model::JuMP.Model, set::ProgrammeAllocationSet,
                                   cbr::ResolvedConstraintBlock, prefix::Symbol)::Nothing
    set_block_constraints!(model, cbr, set, nothing, nothing;
                           tr = allocation_set_tracking(set.tr, cbr), prefix = prefix)
    set_allocation_risk_return_bounds!(model, set, cbr; prefix = prefix)
    set_block_post_risk_constraints!(model, cbr, set, nothing; prefix = prefix)
    return nothing
end
"""
    allocation_set_tracking(tr::Nothing, cbr::ResolvedConstraintBlock)
    allocation_set_tracking(tr, cbr::ResolvedConstraintBlock)

Binds the tracking errors of a programme set to the rows of its block.

The set comes after `factory(set, w)`, so a benchmark that is not fixed already holds the Price-Adjusted Allocation of the step. The function cuts each benchmark series to the row count of the prior of the block with [`prefix_tracking_benchmark`](@ref), and binds each [`DataBenchmarkTracking`](@ref) to the benchmark of the rows `cbr.rd` with [`bind_data_benchmark`](@ref). The constructor of the set refuses a data benchmark under a prior of the Excess Frame, so a data benchmark binds to the raw rows that the set's prior reads.

# Arguments

  - `tr`: The tracking errors of the set, or `nothing`.
  - `cbr`: The Result of the block of the set.

# Returns

  - `tr'`: The bound tracking errors, or `nothing`.

# Related

  - [`add_allocation_set_block!`](@ref)
  - [`prefix_tracking_benchmark`](@ref)
"""
function allocation_set_tracking(::Nothing, ::ResolvedConstraintBlock)::Nothing
    return nothing
end
function allocation_set_tracking(tr, cbr::ResolvedConstraintBlock)
    return bind_data_benchmark(prefix_tracking_benchmark(tr, size(cbr.pr.X, 1)), cbr.rd)
end
"""
    prefix_tracking_benchmark(tr, T::Integer)
    prefix_tracking_benchmark(tr::TrackingError{<:ReturnsTracking}, T::Integer)
    prefix_tracking_benchmark(trs::AbstractVector, T::Integer)

Cuts a [`ReturnsTracking`](@ref) benchmark of a programme set to its first `T` entries when it is longer.

The head's rows are the first rows of the fold, up to the current period, so a benchmark series stated over the whole fold aligns with them by its first `T` entries. A benchmark shorter than the rows goes to the builder unchanged, and [`tracking_benchmark`](@ref) refuses it with an `ArgumentError` that names the two lengths. A weights benchmark reads the rows themselves and is returned unchanged. A vector of tracking errors is cut entry by entry.

# Arguments

  - `tr`: A tracking error, a vector of them, or any other value.
  - `T`: The row count of the head's rows.

# Returns

  - `tr'`: `tr` with each returns benchmark cut to `T` entries.

# Related

  - [`allocation_set_tracking`](@ref)
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
    set_allocation_return_floor!(model::JuMP.Model, ret::Nothing, pr; prefix::Symbol = Symbol(""), kwargs...)
    set_allocation_return_floor!(model::JuMP.Model, ret::JuMPReturnsEstimator, pr::AbstractPriorResult; prefix::Symbol = Symbol(""), kwargs...)
    set_allocation_return_floor!(model::JuMP.Model, rets::VecJRE, pr::AbstractPriorResult; prefix::Symbol = Symbol(""), kwargs...)

Adds a programme set's return floor.

For each term, the per-index arm of [`set_return_constraints!`](@ref) writes the return expression on the prior's expected returns and the row of its `settings.lb`. The term clears its `rte` first with [`no_return_expr_returns_estimator`](@ref), as a risk ceiling clears its `rke`, so it adds nothing to `ret_vec`, the return expression of the objective of the owner. The terms take the indices one to the number of terms under `prefix`, so in a leader's model they do not replace the head's own. The function does not use a term in the objective, as a scalarised return or the normalisation of a ratio, because the objective of the projection is the geometry's divergence.

**Breaking change.** The floor kept `rte`, so it pushed its expression into `ret_vec` of a leader's model. The leader had already collapsed `ret_vec` into its `ret`, so no number changed, but a later reader of `ret_vec` read the term of the set.

# Arguments

  - $(arg_dict[:model])
  - `ret`: The return floor, one return estimator, a vector of them, or `nothing`.
  - `pr`: The set's prior result.
  - $(arg_dict[:prefix_kw])
  - `kwargs...`: Forwarded to [`set_return_constraints!`](@ref). The rows that the head holds arrive as `rd`.

# Returns

  - `nothing`.

# Related

  - [`set_return_constraints!`](@ref)
  - [`set_allocation_risk_return_bounds!`](@ref)
"""
function set_allocation_return_floor!(::JuMP.Model, ::Nothing, ::Any;
                                      prefix::Symbol = Symbol(""), kwargs...)::Nothing
    return nothing
end
function set_allocation_return_floor!(model::JuMP.Model, ret::JuMPReturnsEstimator,
                                      pr::AbstractPriorResult; prefix::Symbol = Symbol(""),
                                      kwargs...)::Nothing
    set_allocation_return_floor!(model, [ret], pr; prefix = prefix, kwargs...)
    return nothing
end
function set_allocation_return_floor!(model::JuMP.Model, rets::VecJRE,
                                      pr::AbstractPriorResult; prefix::Symbol = Symbol(""),
                                      kwargs...)::Nothing
    for (i, ret) in enumerate(rets)
        set_return_constraints!(model, i, no_return_expr_returns_estimator(ret), pr;
                                prefix = prefix, kwargs...)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds the weight bounds and the budget of one of a [`BoundedAllocationSet`](@ref) to a model that already names its own.

The rows are the inequalities that [`set_allocation_set_bounds!`](@ref) writes, registered under the prefix `:aset_` as `aset_w_lb`, `aset_w_ub` and `aset_bgt`. A [`ProgrammeAllocationSet`](@ref) does not come here: the block builder [`set_block_constraints!`](@ref) adds its bounds and its short and gross budgets under its own prefix.

**Breaking change.** The rows were anonymous. They are registered now, so a reader of the model can find them.

# JuMP formulation

## Variables

  - `w`, `k`: read from the model.

## Constraints

  - `aset_w_lb`: ``s_c (\\boldsymbol{w} - k \\boldsymbol{l}) \\geq 0``, when an entry of ``\\boldsymbol{l}`` is finite.
  - `aset_w_ub`: ``s_c (\\boldsymbol{w} - k \\boldsymbol{u}) \\leq 0``, when an entry of ``\\boldsymbol{u}`` is finite.
  - `aset_bgt`: ``s_c \\left(\\sum_{i=1}^{N} w_i - k\\right) = 0``.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:k_budget])
  - $(math_dict[:w_port])
  - $(math_dict[:N])
  - ``\\boldsymbol{l}``, ``\\boldsymbol{u}``: Lower and upper weight bounds of the set, resolved.

# Arguments

  - $(arg_dict[:model])
  - `wb`: The set's resolved [`WeightBounds`](@ref).

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub` over the resolved bounds, through [`assert_feasible_bounds`](@ref).

# Related

  - [`add_allocation_set_constraints!`](@ref)
  - [`set_allocation_set_bounds!`](@ref)
"""
function add_allocation_set_bounds!(model::JuMP.Model, wb::WeightBounds)::Nothing
    assert_feasible_bounds(wb)
    state_build!(() -> get_w(model), model, :aset_, :w)
    set_weight_constraints!(model, wb, nothing, nothing; prefix = :aset_)
    set_budget_constraints!(model, 1, get_w(model); prefix = :aset_)
    return nothing
end
"""
    set_allocation_set_bounds!(model::JuMP.Model, wb::WeightBounds)

Adds the weight bounds and the budget of one of a [`BoundedAllocationSet`](@ref) to the bare projection model.

[`set_weight_constraints!`](@ref) writes the bounds with no net budget, and the function adds `Σw = k` after that, through [`set_budget_constraints!`](@ref). A [`ProgrammeAllocationSet`](@ref) does not come here: the block builder [`set_block_constraints!`](@ref) adds its bounds and its short and gross budgets.

# Algorithm

 1. Check the bounds against the budget with [`assert_feasible_bounds`](@ref).
 2. Add the bounds with [`set_weight_constraints!`](@ref), with no net budget.
 3. Add the budget `Σw = k` with [`set_budget_constraints!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `wb`: The set's resolved [`WeightBounds`](@ref).

# Validation

  - `Σ lb ≤ 1 ≤ Σ ub` over the resolved bounds, through [`assert_feasible_bounds`](@ref).

# Related

  - [`set_allocation_set_constraints!`](@ref)
  - [`BoundedAllocationSet`](@ref)
"""
function set_allocation_set_bounds!(model::JuMP.Model, wb::WeightBounds)::Nothing
    assert_feasible_bounds(wb)
    set_weight_constraints!(model, wb, nothing, nothing)
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
  - Under [`GramProjection`](@ref): `proj.A` is positive definite. A `LinearAlgebra.PosDefException` is thrown otherwise. A singular Gram matrix states a seminorm, in which the projection is not unique. [`NewtonStep`](@ref) binds ``\\mathbf{I} + \\sum_{s} \\boldsymbol{g}_s \\boldsymbol{g}_s^\\intercal``, which is positive definite.

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
    P = length(pos)
    JuMP.@variable(model, t_proj[1:P])
    # `t_i ≤ w_i^α` is the power cone `w_i^α · 1^(1 − α) ≥ |t_i|`.
    JuMP.@constraint(model, [i = 1:P],
                     [sc * w[pos[i]], sc, sc * t_proj[i]] in JuMP.MOI.PowerCone(a))
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

The penalty holds the set's penalties, its custom objective terms and the `p · tr(W)` term of its semidefinite phylogeny. It enters through [`add_penalty_to_objective!`](@ref), which every JuMP head's objective also uses. With no penalty the objective is the divergence alone, which is the plain projection of every published rule.

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

  - Under no name, one vector row: ``s_c w_i = 0`` for each ``i`` with ``q_i = 0``. A `q` with no zero entry writes no row.

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
    zs = findall(iszero, q)
    if !isempty(zs)
        JuMP.@constraint(model, sc * view(w, zs) == 0)
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

A set that reads the rows fits its prior on the rows of the step, as a batch head does. The programme then runs on the prior's Investable Mask, also as a batch head does. It solves over the assets the prior prices, and gives a zero weight to every other asset. Under a time-varying panel, an asset without a listing for part of the window is outside the universe of a plain prior until the window no longer holds that part. So the programme gives it a zero, which is the one zero in the recursion's allocation that the rule's step did not make. A batch head gives the same zero. The asset comes back at the first step at which the prior prices it, and the rule's geometry decides whether the raw step then gives it weight, as for a zero in the Start Allocation. A set that reads no rows fits no prior and solves over the full universe.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{w}^{+} &= \\underset{\\boldsymbol{w} \\in \\mathcal{W}}{\\arg\\min} \\; D_\\Psi(\\boldsymbol{w}, \\boldsymbol{q}) + \\pi(\\boldsymbol{w})\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_plus_proj])
  - $(math_dict[:W_aset]) Under an Investable Mask it also holds ``w_i = 0`` for every asset ``i`` outside the mask.
  - $(math_dict[:D_Psi_breg])
  - $(math_dict[:Psi_pot])
  - $(math_dict[:q_raw])
  - $(math_dict[:pi_obj_pen])

[`set_projection_objective!`](@ref) states the objective each geometry writes for ``D_\\Psi``.

# Algorithm

 1. Read the rows `X` of the current [`ProjectionStep`](@ref).
 2. When [`allocation_set_ready`](@ref) gives a reason, record the Held Step with it and return a copy of `w`. Stop.
 3. Fit the set's prior `pr` on `X` with [`allocation_set_prior`](@ref), and read its Investable Mask `imsk` with [`prior_investable_mask`](@ref).
 4. Build and solve the programme at `imsk` with the seven-argument method, and return its value. It reduces the set, the rows, the prior, the geometry, `q` and `w` to the mask, adds the set's constraints, the custom objective terms and the objective, and solves on [`projection_solver`](@ref) with [`solve_projection_programme!`](@ref), which solves a second stage only under a geometry whose divergence has many minimisers. A distance reader of the set that refuses an asset makes it build again without that asset. A solve that gives an allocation returns it with a zero at every asset outside the mask. Otherwise the method records the Held Step with the solver trials, and returns a copy of `w`.

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
    return projection_programme(proj, set, q, w, X, pr, prior_investable_mask(pr))
end
"""
    projection_programme(proj::AbstractProjectionGeometry, set::AbstractAllocationSet,
                         q::AbstractVector, w::AbstractVector, X::Option{<:ReturnsResult},
                         pr::Option{<:AbstractPriorResult}, imsk::Option{BitVector})

Builds and solves the projection programme of one step at the Investable Mask `imsk`, from the prior result `pr` that the step fitted once.

A distance reader of the set, its `ple` or its `cte`, can refuse an asset that the prior keeps, see [`DistanceReaderRefusalError`](@ref). The programme learns of the refusal only when it adds the set's constraints, after the reduction. So the method catches the refusal around [`set_allocation_set_constraints!`](@ref) with [`distance_reader_call`](@ref), composes the mask with [`distance_reader_refusal_mask`](@ref), and builds the model again at that mask. The prior is not fitted again, so the step reads the same moments. This is the rule of a batch JuMP head. The refused asset takes a zero weight at the step, as an asset that the prior cannot price does, and the warning of [`distance_reader_refusal_mask`](@ref) names it. The Investable Mask of the Recursion Result does not change, because it is the active mask of the last row. Under the `strict` flag of the step the method refuses the asset. The two readers can refuse different assets, and the build again meets the refusal of the other one.

# Algorithm

 1. Reduce `set`, `X` and `pr` to `imsk` with [`programme_investable_reduction`](@ref). Reduce the geometry and `q` to the mask with [`investable_geometry_view`](@ref), and view `w` at the mask, which gives `wr`.
 2. Make a bare model with the set's two scales and the row count of `X`, zero when there are no rows. Register `k = 1`, and the vector `w` of weight variables.
 3. Add the set's constraints with [`set_allocation_set_constraints!`](@ref) through [`distance_reader_call`](@ref), with `wr` as the reference. This gives the weight bounds `wb` of the step and the Result `cbr` of the block of a programme set. When a distance reader refuses an asset, call this method again with the mask of [`distance_reader_refusal_mask`](@ref) under [`projection_step_strict`](@ref), and return its value.
 4. Refuse a floor of `wb` that the geometry does not admit, with [`assert_geometry_admits_bounds`](@ref).
 5. Add the set's custom objective terms with [`add_custom_objective_term!`](@ref), with the geometry as the objective, the set as the owner, and `cbr` as the bundle that the terms read.
 6. Set the objective with [`set_projection_objective!`](@ref).
 7. Solve on [`projection_solver`](@ref) with [`solve_projection_programme!`](@ref). When the solve gives an allocation, take its answer on the mask with [`programme_answer`](@ref), and return it with a zero at every asset outside the mask. Otherwise record the Held Step with the solver trials, and return a copy of `w`.

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
  - `X`: The rows of the step, or `nothing`.
  - `pr`: The set's prior result on `X`, or `nothing` when the set reads no rows.
  - `imsk`: The Investable Mask of the step, or `nothing` when every asset is investable.

# Validation

  - Under the `strict` flag of the step, no distance reader refuses an asset. Raises an [`IsNonFiniteError`](@ref).
  - The rules of [`assert_geometry_admits_bounds`](@ref) on the bounds of the step.
  - At least one asset stays. Raises an [`IsEmptyError`](@ref).

# Returns

  - `w'::Vector`: The projected allocation, or `w` copied on a Held Step.

# Related

  - [`distance_reader_refusal_mask`](@ref)
  - [`distance_reader_call`](@ref)
  - [`programme_investable_reduction`](@ref)
  - [`resolve_jump_optimiser`](@ref)
"""
function projection_programme(proj::AbstractProjectionGeometry, set::AbstractAllocationSet,
                              q::AbstractVector, w::AbstractVector,
                              X::Option{<:ReturnsResult}, pr::Option{<:AbstractPriorResult},
                              imsk::Option{BitVector})
    setr, Xr, prr = programme_investable_reduction(imsk, set, X, pr)
    projr, qr = investable_geometry_view(imsk, proj, q)
    wr = investable_weights_view(imsk, w)
    model = JuMP.Model()
    set_model_scales!(model, projection_scale(setr, :sc), projection_scale(setr, :so))
    set_model_observations!(model, isnothing(Xr) ? 0 : size(Xr.X, 1))
    JuMP.@expression(model, k, 1)
    wv = state_set!(model, Symbol(""), :w,
                    JuMP.@variable(model, [1:length(qr)], base_name = "w"))
    res = distance_reader_call(set_allocation_set_constraints!, model, setr, wr, Xr, prr)
    if res isa DistanceReaderRefusalError
        # Only a set that reads rows calls a reader, so a refusal comes with rows.
        return projection_programme(proj, set, q, w, X, pr,
                                    distance_reader_refusal_mask(res, imsk,
                                                                 X::ReturnsResult,
                                                                 projection_step_strict()))
    end
    wb, cbr = res
    assert_geometry_admits_bounds(projr, wb)
    # A block's custom objective terms get the objective, the owner of the block and the
    # Result of the block (ADR 0237).
    add_custom_objective_term!(model, projr, allocation_set_cobj(cbr), setr, cbr)
    set_projection_objective!(model, projr, qr)
    res, wn = solve_projection_programme!(model, projr, qr, projection_solver(projr, setr))
    if !isnothing(wn)
        return expand_investable_weights(imsk, programme_answer(projr, wn, wb))
    end
    record_held_step!("the projection onto the `$(nameof(typeof(set)))` in the `$(nameof(typeof(proj)))` geometry did not solve, and the step trades nothing",
                      res.trials)
    return copy(w)
end
"""
    programme_investable_reduction(imsk::Nothing, set::AbstractAllocationSet, X::Option{<:ReturnsResult}, pr)
    programme_investable_reduction(imsk::BitVector, set::AbstractAllocationSet, X::ReturnsResult, pr::Option{<:AbstractPriorResult})
    programme_investable_reduction(imsk::BitVector, set::AbstractAllocationSet, X, pr)

Reduces the set, the rows that the head holds and the set's prior to the prior's Investable Mask.

The projection programme then runs on the assets the prior can price, as a batch head does with [`investable_reduction`](@ref). The caller views the raw step and the Price-Adjusted Allocation at the same mask with [`investable_weights_view`](@ref). The entry of a JuMP head reduces the block of a set to the mask of every block in the same way, [`resolve_allocation_set_block`](@ref).

Dispatch selects one of three methods. A `nothing` mask means that every asset is investable, which is true at every step on a static panel and at every step of a set that reads no rows. That method returns its arguments unchanged. The `BitVector` method takes the views at `findall(imsk)`. A set that reads no rows has no prior, and the method views the set and the rows alone. This pair occurs at the entry of a JuMP head, whose own prior gives the mask. The third method refuses a mask with no rows, because a mask is read on the rows. The set is viewed against the full returns matrix of the rows, because a view of a tracking estimator needs it. After the view, the method declares the names outside the mask on the Non-Investable Axis of the set's `sets`, with [`non_investable_universe`](@ref), as [`investable_reduction`](@ref) does for a head. So a row of `lcse` or `gcarde` that names such an asset drops at stage 2 of the block, and `strict` does not refuse the name. The Price-Adjusted Allocation is cut to the mask and is not renormalised, as the batch reduction cuts a turnover reference. So an asset the prior cannot price leaves the reference, and the programme allocates the budget of one over the investable assets. The function writes no warning for such an asset, because the Investable Mask of the Recursion Result already shows it at every step.

# Validation

  - A mask with no rows. An `IsNothingError` is thrown.

# Returns

  - `(set, X, pr)::Tuple`: The three reduced to the mask, or unchanged. `pr` stays `nothing` when the set has no prior.

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
                                        X::ReturnsResult, pr::Option{<:AbstractPriorResult})
    idx = findall(imsk)
    # The view drops the Non-Investable Axis, so the set declares it after the view, as
    # `investable_reduction` declares it for a head. Stage 2 of the block then drops a row
    # that names a departed asset, and does not refuse the name.
    return non_investable_universe(port_opt_view(set, idx, X.X),
                                   non_investable_names(X.nx, imsk)), port_opt_view(X, idx),
           port_opt_view(pr, idx)
end
function programme_investable_reduction(::BitVector, ::AbstractAllocationSet, ::Any, ::Any)
    return throw(IsNothingError("an Investable Mask is read on the rows that the head holds, so a mask with no rows cannot be reduced on."))
end
"""
    investable_geometry_view(imsk::Nothing, proj::AbstractProjectionGeometry, q::AbstractVector)
    investable_geometry_view(imsk::BitVector, proj::AbstractProjectionGeometry, q::AbstractVector)
    investable_geometry_view(imsk::BitVector, proj::DiagonalProjection, q::AbstractVector)
    investable_geometry_view(imsk::BitVector, proj::GramProjection{<:Any, <:AbstractMatrix}, q::AbstractVector)

Reduces the Projection Geometry and the raw step `q` to the Investable Mask of a projection programme.

[`projection_programme`](@ref) solves over the assets in the mask, and gives a zero weight to every other asset. A `nothing` mask returns both arguments unchanged. A geometry whose divergence is a sum of one term for each asset needs only the view of `q`, because the term of an asset outside the mask is a constant at its zero weight. [`DiagonalProjection`](@ref) is such a sum, and its diagonal is viewed with `q`. The norm of a [`GramProjection`](@ref) couples the assets, so the zero weights move the point that the reduced programme projects.

# Mathematical definition

Let ``P`` be the assets in the mask and ``Z`` the others, with ``\\boldsymbol{w}_Z = \\boldsymbol{0}``. Then

```math
\\begin{align}
(\\boldsymbol{w} - \\boldsymbol{q})^\\intercal \\mathbf{A} (\\boldsymbol{w} - \\boldsymbol{q}) &= (\\boldsymbol{w}_P - \\tilde{\\boldsymbol{q}}_P)^\\intercal \\mathbf{A}_{PP} (\\boldsymbol{w}_P - \\tilde{\\boldsymbol{q}}_P) + c\\,, \\\\
\\tilde{\\boldsymbol{q}}_P &= \\boldsymbol{q}_P + \\mathbf{A}_{PP}^{-1} \\mathbf{A}_{PZ} \\boldsymbol{q}_Z\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:q_raw])
  - ``\\mathbf{A}``: Gram matrix of the geometry, positive definite.
  - ``\\mathbf{A}_{PP}``, ``\\mathbf{A}_{PZ}``: Blocks of ``\\mathbf{A}`` at the rows ``P`` and the columns ``P`` and ``Z``. ``\\mathbf{A}_{PP}`` is positive definite, because ``\\mathbf{A}`` is.
  - ``\\tilde{\\boldsymbol{q}}_P``: Raw step of the reduced programme.
  - ``c``: Term that does not depend on ``\\boldsymbol{w}_P``.

So the reduced Gram geometry holds ``\\mathbf{A}_{PP}`` and projects ``\\tilde{\\boldsymbol{q}}_P``. The Gram method reads a bound matrix. [`NewtonStep`](@ref), the one rule that takes a [`GramProjection`](@ref), binds the matrix before the view through [`gram_geometry`](@ref).

# Returns

  - `(proj, q)::Tuple`: The geometry and the raw step on the assets in the mask.

# Related

  - [`projection_programme`](@ref)
  - [`programme_investable_reduction`](@ref)
  - [`investable_weights_view`](@ref)
  - [`DiagonalProjection`](@ref)
  - [`GramProjection`](@ref)
"""
function investable_geometry_view(::Nothing, proj::AbstractProjectionGeometry,
                                  q::AbstractVector)
    return proj, q
end
function investable_geometry_view(imsk::BitVector, proj::AbstractProjectionGeometry,
                                  q::AbstractVector)
    return proj, investable_weights_view(imsk, q)
end
function investable_geometry_view(imsk::BitVector, proj::DiagonalProjection,
                                  q::AbstractVector)
    return DiagonalProjection(proj.h[imsk]), investable_weights_view(imsk, q)
end
function investable_geometry_view(imsk::BitVector,
                                  proj::GramProjection{<:Any, <:AbstractMatrix},
                                  q::AbstractVector)
    A = proj.A
    zmsk = .!imsk
    App = A[imsk, imsk]
    qp = q[imsk] .+ LinearAlgebra.Symmetric(App) \ (A[imsk, zmsk] * q[zmsk])
    return GramProjection(; slv = proj.slv, A = App), qp
end
"""
    solve_projection_programme!(model::JuMP.Model, proj::AbstractProjectionGeometry, q::AbstractVector, slv::Slv_VecSlv)
    solve_projection_programme!(model::JuMP.Model, proj::DiagonalProjection, q::AbstractVector, slv::Slv_VecSlv)

Solves a projection programme, and picks one minimiser when the divergence of the geometry has many.

[`projection_programme`](@ref) calls it after it writes the objective. Every geometry except [`DiagonalProjection`](@ref) has one minimiser, and the function solves the model once with [`optimise_JuMP_model!`](@ref). A diagonal with a zero entry is a seminorm, so its minimisers share their weights at the positive entries of the diagonal and differ at the zero entries. After a first solve that succeeds, the diagonal arm then solves a second stage on the same model. Its answer is the limit of the projection in ``\\mathrm{diag}(\\boldsymbol{h} + \\delta \\boldsymbol{1})`` as ``\\delta \\to 0^{+}``: the minimiser nearest ``\\boldsymbol{q}`` in the Euclidean norm.

When the second stage fails, the arm returns the weights of the first stage, and the result is a success that holds the trials of the second stage. Those weights are a minimiser of the projection, so the regret bound of [`AdaptiveSubgradient`](@ref) holds on them. The second stage can fail when its face is one point: under a strictly convex Objective Penalty, or when the budget pins the free weights at their bounds. That point is then the limit. On a face of more than one point, the weights of the first stage are a minimiser that need not be the limit.

# JuMP formulation

The second stage of the diagonal arm, when ``Z`` is not empty.

## Variables

  - `w`: read from the model.
  - `t_sel`: created.

## Constraints

  - `proj_face`: ``s_o \\, (d + \\pi) \\leq v``, when the model holds an Objective Penalty. It keeps the objective of the first stage at its minimum, so the penalty stays at its minimum too. The point of the first stage meets it, so the second stage is feasible to the tolerance of the solver. A slack would let the free weights move along the curve of the penalty by the square root of the slack, so the bound has none.
  - `proj_fix`: ``s_c (w_i - w_i^{(1)}) = 0`` for each ``i \\in P``.
  - `proj_sel`: ``(s_c t, s_c (\\boldsymbol{w}_Z - \\boldsymbol{q}_Z)) \\in \\mathcal{K}_{\\mathrm{SOC}}``.

## Objective

  - `Min`: ``s_o t``.

Where:

  - ``P = \\{i : h_i > 0\\}``, ``Z = \\{i : h_i = 0\\}``: Assets at the positive and at the zero entries of the diagonal.
  - ``w_i^{(1)}``: Weight of asset ``i`` at the first stage, which every minimiser shares.
  - ``v``: Objective value of the first stage.
  - ``d + \\pi``: The expression `obj_expr` of the first stage, the divergence and the Objective Penalty.
  - ``t``: The variable `t_sel`.
  - $(math_dict[:sc_scale])
  - $(math_dict[:so_scale])
  - ``\\mathcal{K}_{\\mathrm{SOC}}``: Second-order cone.

# Arguments

  - $(arg_dict[:model])
  - `proj`: The geometry, reduced to the Investable Mask.
  - `q`: The raw step, reduced to the Investable Mask.
  - `slv`: The solver of the programme.

# Returns

  - `(res, w)::Tuple`: `res::JuMPResult` is the result of the last solve, and a first solve that fails is the last one. `w::Option{<:AbstractVector}` is the solved allocation, or `nothing` when no solve gives one.

# Related

  - [`projection_programme`](@ref)
  - [`DiagonalProjection`](@ref)
  - [`set_projection_objective!`](@ref)
  - [`optimise_JuMP_model!`](@ref)
"""
function solve_projection_programme!(model::JuMP.Model, ::AbstractProjectionGeometry,
                                     ::AbstractVector, slv::Slv_VecSlv)
    res = optimise_JuMP_model!(model, slv)
    return res, res.success ? JuMP.value.(get_w(model)) : nothing
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
    allocation_set_cobj(cbr::ResolvedConstraintBlock)
    allocation_set_cobj(cbr::Nothing)

Returns the custom objective terms that a projection programme adds to its objective.

A programme set gives the `cobj` of its [`ResolvedConstraintBlock`](@ref), which holds the terms after `factory(set, w)`. A bounded set has no block Result and carries no objective term, so it gives `nothing`.

# Related

  - [`projection_programme`](@ref)
  - [`add_custom_objective_term!`](@ref)
"""
function allocation_set_cobj(cbr::ResolvedConstraintBlock)
    return cbr.cobj
end
function allocation_set_cobj(::Nothing)
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

These are the programme arms of the Constrained Update. They cover every pair of a geometry and a set except the scalar roots on the bounded set. Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref), [`programme_answer`](@ref) sets every entry of the answer within the tolerance of its element type to zero. The solver gives a weight that the set forces to zero slightly above or slightly below zero, and the next step must read it as zero for either sign. A zero then stays zero at every later step, because the divergence from a zero entry is infinite. When the set later needs that asset, the step is a Held Step.

# Validation

  - Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref): `all(>= 0, lb)` over the bounds of the step, through [`assert_geometry_admits_bounds`](@ref) in [`projection_programme`](@ref). A `DomainError` is thrown otherwise.

# Related

  - [`project`](@ref)
  - [`projection_programme`](@ref)
  - [`programme_answer`](@ref)
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
function project(proj::Union{EntropicProjection, GramProjection, TsallisProjection,
                             LogBarrierProjection}, set::ProgrammeAllocationSet,
                 q::AbstractVector, w::AbstractVector)
    return projection_programme(proj, set, q, w)
end
"""
    programme_answer(proj::AbstractProjectionGeometry, w::AbstractVector, wb::WeightBounds)

Returns the answer of a solved projection programme, on the assets of the step.

Under [`EntropicProjection`](@ref), [`TsallisProjection`](@ref) and [`LogBarrierProjection`](@ref), [`clip_at_zero`](@ref) sets every entry within the tolerance of its element type to its floor in `wb`. Every other geometry returns `w`. The floor is the floor of the bounds of the step, which [`set_allocation_set_constraints!`](@ref) returns, so a bound that reads the rows clips at its own floor.

# Arguments

  - `proj`: The geometry, viewed at the Investable Mask.
  - `w`: The weights of the solve, on the assets of the mask.
  - `wb`: The weight bounds of the step, on the same assets.

# Returns

  - `w'::AbstractVector`: The answer on the assets of the mask.

# Related

  - [`projection_programme`](@ref)
  - [`clip_at_zero`](@ref)
"""
function programme_answer(::AbstractProjectionGeometry, w::AbstractVector, ::WeightBounds)
    return w
end
function programme_answer(::Union{EntropicProjection, TsallisProjection,
                                  LogBarrierProjection}, w::AbstractVector,
                          wb::WeightBounds)
    return clip_at_zero(w, something(wb.lb, zero(eltype(w))))
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
