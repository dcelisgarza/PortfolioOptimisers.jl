# =============================================================================
# Prototype 15 — Explicit portfolio state.
#
# Purpose
#   Report 4 identifies this as the strongest lesson from OptimalPortfolios,
#   and states it in one sentence:
#
#       portfolio state should be explicit rather than treating optimisation
#       as a stateless function of returns.
#
#   Three concrete consequences, each of which the library gets wrong today by
#   omission rather than by error:
#
#     1. **Drift.** Between two rebalances the held weights move with the
#        market. Turnover measured against the *previous target* rather than
#        against the *drifted holdings* is wrong, and it is wrong in the
#        direction that understates trading.
#     2. **Tradability.** A real portfolio cannot trade every name at every
#        rebalance. Frozen and unavailable positions must be held while the
#        rest is optimised over the remaining budget.
#     3. **A changing universe.** An asset with three months of history cannot
#        enter a model that needs a year of it, and the answer must still be
#        expressed over the full universe.
#
# Status
#   Standalone. Depends on `LinearAlgebra` and `Statistics`.
#
# Notation used throughout this file
#   N        Number of assets in the full universe.
#   w        Held weights over the full universe, length `N`. They sum to the
#            invested fraction, which is one minus cash.
#   r        Realised simple returns over the holding period, length `N`.
#   status   Per-asset tradability, length `N`.
#   budget   Total weight the portfolio must sum to.
#
# Sources
#   Grinold, R. C. and Kahn, R. N. (1999). Active Portfolio Management, 2nd
#     edition. McGraw-Hill. Chapter 16 on implementation and the difference
#     between a paper portfolio and a held one.
#   Perold, A. F. (1988). The implementation shortfall: paper versus reality.
#     Journal of Portfolio Management 14(3), 4-9. The original statement that
#     the gap between the model portfolio and the held one is itself a cost.
#   Almgren, R. and Chriss, N. (2001). Optimal execution of portfolio
#     transactions. Journal of Risk 3(2), 5-39. See prototype 17.
# =============================================================================
module PortfolioState

using LinearAlgebra, Statistics

export AssetStatus, TRADABLE, FROZEN, UNAVAILABLE, NEW, HeldPortfolio, drift_weights,
       realised_turnover, tradable_subproblem, restore_universe, eligible_by_history,
       relax_group_bound

"""
    AssetStatus

Tradability of a single asset at a rebalance.

  - `TRADABLE`: may be bought or sold freely.
  - `FROZEN`: currently held and **must be kept at its drifted weight**. A
    private holding, a locked-up position, or a name whose sale would realise
    an unacceptable tax charge.
  - `UNAVAILABLE`: cannot be held at all. Suspended, delisted, or outside the
    permitted universe. Its weight must be zero.
  - `NEW`: eligible but not yet held, and with too little history to estimate.
    Treated as tradable with a zero starting weight, and excluded from any
    estimator that needs a full window.

# Notes

  - **`FROZEN` and `UNAVAILABLE` are not opposites.** A frozen asset is held
    and cannot be sold. An unavailable one is not held and cannot be bought. A
    position that is both held and suspended is the hard case, and it must be
    modelled as `FROZEN`, because the weight is real whatever the exchange
    says.
"""
@enum AssetStatus TRADABLE FROZEN UNAVAILABLE NEW

"""
    HeldPortfolio{T}

The state of a portfolio at the moment before a rebalance.

# Fields

  - `w::Vector{T}`: Held weights over the full universe, length `N`. These are
    the **drifted** weights, not the last target.
  - `status::Vector{AssetStatus}`: Tradability, length `N`.
  - `cash::T`: Uninvested fraction. `sum(w) + cash` is one.
  - `names::Vector{String}`: Asset labels, length `N`.

# Validation

  - All four fields agree on `N`.
  - An `UNAVAILABLE` asset has zero weight.
"""
struct HeldPortfolio{T <: Real}
    w::Vector{T}
    status::Vector{AssetStatus}
    cash::T
    names::Vector{String}
    function HeldPortfolio(w::Vector{T}, status::Vector{AssetStatus}, cash::T,
                           names::Vector{String}) where {T <: Real}
        N = length(w)
        if length(status) != N || length(names) != N
            throw(DimensionMismatch("w, status and names must all have length $(N)"))
        end
        for i in 1:N
            if status[i] === UNAVAILABLE && !iszero(w[i])
                throw(ArgumentError("asset $(names[i]) is UNAVAILABLE but holds weight $(w[i])"))
            end
        end
        return new{T}(w, status, cash, names)
    end
end
function HeldPortfolio(w::AbstractVector{<:Real};
                       status::AbstractVector{AssetStatus} = fill(TRADABLE, length(w)),
                       cash::Real = 1 - sum(w),
                       names::AbstractVector{<:AbstractString} = ["asset $(i)"
                                                                  for i in 1:length(w)])
    wv = collect(float.(w))
    return HeldPortfolio(wv, collect(status), eltype(wv)(cash), String.(collect(names)))
end

"""
    drift_weights(w::AbstractVector, r::AbstractVector; renormalise::Bool = true)
        -> Vector

Return the weights after a holding period, before any trade.

# Arguments

  - `w`: Weights at the start of the period, length `N`.
  - `r`: Realised simple returns over the period, length `N`.
  - `renormalise`: If `true`, rescale so the result sums to the same total as
    `w`. If `false`, return the raw grown weights, whose sum is the portfolio's
    gross return times the original budget.

# Returns

  - The drifted weights, length `N`.

# Mathematical definition

    w_drift_i  =  w_i (1 + r_i)  /  sum_j w_j (1 + r_j)   *  sum_j w_j

**Nobody rebalances continuously**, so this is what is actually held when the
next optimisation runs. The difference from `w` is free: it happened without
trading.

# Notes

  - **`renormalise = false` is the honest choice when cash is held**, because
    cash does not grow with the market and renormalising against the invested
    total silently reallocates the gain. Set it to `false` and track cash
    separately whenever `sum(w) < 1`.
"""
function drift_weights(w::AbstractVector{<:Real}, r::AbstractVector{<:Real};
                       renormalise::Bool = true)
    if length(w) != length(r)
        throw(DimensionMismatch("w has length $(length(w)), r has length $(length(r))"))
    end
    grown = collect(float.(w)) .* (one(float(eltype(w))) .+ r)
    if !renormalise
        return grown
    end
    s = sum(grown)
    return iszero(s) ? grown : grown .* (sum(w) / s)
end

"""
    realised_turnover(w_target::AbstractVector, state::HeldPortfolio) -> NamedTuple

Return the turnover a rebalance actually incurs, and the figure a
state-blind calculation would have reported.

# Arguments

  - `w_target`: The new target weights, length `N`.
  - `state`: The current state, whose `w` field holds the **drifted** weights.

# Returns

A `NamedTuple` with `turnover` (against the drifted holdings), `by_asset`, and
`traded_names`.

# Mathematical definition

    turnover  =  (1/2) sum_i | w_target_i  -  w_drift_i |

# Notes

  - **The comparison must be against what is held, not against what was last
    targeted.** A portfolio that drifted towards its new target trades less
    than the naive figure says; one that drifted away trades more. Both errors
    are real money, and both are invisible to a calculation that never sees the
    holdings.
  - The library's turnover machinery in `src/15_Turnover.jl` takes a previous
    weight vector, so it is *capable* of this. What is absent is the state
    object that makes the drifted vector the natural thing to pass.
"""
function realised_turnover(w_target::AbstractVector{<:Real}, state::HeldPortfolio)
    N = length(state.w)
    if length(w_target) != N
        throw(DimensionMismatch("w_target has length $(length(w_target)), state has $(N) assets"))
    end
    d = abs.(collect(float.(w_target)) .- state.w)
    traded = [state.names[i] for i in 1:N if d[i] > 1e-10]
    return (; turnover = sum(d) / 2, by_asset = d, traded_names = traded)
end

"""
    tradable_subproblem(state::HeldPortfolio; budget::Real = 1.0) -> NamedTuple

Return the reduced problem that the optimiser should actually solve.

# Arguments

  - `state`: The current portfolio state.
  - `budget`: The total weight the full portfolio must sum to.

# Returns

A `NamedTuple`:

  - `idx`: Indices the optimiser may set, those with status `TRADABLE` or
    `NEW`.
  - `held_fixed`: The total weight locked in frozen positions.
  - `sub_budget`: `budget - held_fixed`, the budget for the reduced problem.
  - `frozen_idx`, `unavailable_idx`: The other two groups.

# Notes

  - **The sub-budget is the whole trick.** Optimise the tradable names to sum
    to `budget - held_fixed` and the reassembled portfolio satisfies the
    original budget exactly, with no post-hoc renormalisation. Renormalising
    afterwards would silently scale the frozen positions, which is precisely
    what "frozen" forbids.
  - A negative `sub_budget` means the frozen positions already exceed the
    budget. That is feasible in reality and infeasible for the optimiser, so it
    is raised rather than clamped.
"""
function tradable_subproblem(state::HeldPortfolio; budget::Real = 1.0)
    idx = findall(s -> s === TRADABLE || s === NEW, state.status)
    frozen_idx = findall(==(FROZEN), state.status)
    unavailable_idx = findall(==(UNAVAILABLE), state.status)
    held_fixed = if isempty(frozen_idx)
        zero(eltype(state.w))
    else
        sum(view(state.w, frozen_idx))
    end
    sub = budget - held_fixed
    if sub < 0
        throw(ArgumentError("frozen positions total $(held_fixed), which exceeds the budget $(budget); the problem is infeasible without a sale"))
    end
    return (; idx = idx, held_fixed = held_fixed, sub_budget = sub, frozen_idx = frozen_idx,
            unavailable_idx = unavailable_idx)
end

"""
    restore_universe(w_sub::AbstractVector, sub, state::HeldPortfolio) -> Vector

Expand a solution over the tradable subset back to the full universe.

# Arguments

  - `w_sub`: The optimiser's answer over `sub.idx`.
  - `sub`: The output of [`tradable_subproblem`](@ref).
  - `state`: The state it was derived from.

# Returns

  - Full-universe weights, length `N`, with frozen positions at their held
    values and unavailable positions at zero.

# Notes

  - This is the counterpart to the restriction, and keeping the pair explicit
    is what stops an asset silently vanishing from a report because it was
    untradable for one period.
"""
function restore_universe(w_sub::AbstractVector{<:Real}, sub, state::HeldPortfolio)
    if length(w_sub) != length(sub.idx)
        throw(DimensionMismatch("w_sub has length $(length(w_sub)), the tradable set has $(length(sub.idx))"))
    end
    w = zeros(float(eltype(state.w)), length(state.w))
    for (k, i) in enumerate(sub.idx)
        w[i] = w_sub[k]
    end
    for i in sub.frozen_idx
        w[i] = state.w[i]
    end
    return w
end

"""
    eligible_by_history(X::AbstractMatrix, min_obs::Integer) -> NamedTuple

Return which assets have enough usable history, treating `NaN` and `missing`
as absent.

# Arguments

  - `X`: Returns, `T x N`, possibly with `NaN` entries.
  - `min_obs`: Minimum number of finite observations required.

# Returns

A `NamedTuple` with `eligible` (indices), `n_obs` (finite count per asset) and
`ineligible`.

# Notes

  - **This is the cheapest correct answer, not the best one.** Counting finite
    observations ignores *where* they sit: an asset with 300 finite points all
    before 2015 passes a 250-point test and is still useless. A production rule
    also needs a recency condition. The count is implemented here because it is
    the part every caller needs and nobody writes twice.
  - The library's `MissingDataFilter` handles the price level. What is absent
    is the per-asset eligibility rule at the point of optimisation, which is a
    different decision: a filter drops rows, this drops columns.
"""
function eligible_by_history(X::AbstractMatrix, min_obs::Integer)
    N = size(X, 2)
    counts = [count(x -> !ismissing(x) && isfinite(x), view(X, :, j)) for j in 1:N]
    eligible = findall(>=(min_obs), counts)
    return (; eligible = eligible, n_obs = counts, ineligible = setdiff(1:N, eligible))
end

"""
    relax_group_bound(w_frozen_total::Real, lower::Real, upper::Real;
                      group_name::AbstractString = "group") -> NamedTuple

Widen a group bound just enough to admit the frozen holdings, and record that
it happened.

# Arguments

  - `w_frozen_total`: Weight already locked inside the group.
  - `lower`, `upper`: The intended group bounds.
  - `group_name`: Label used in the audit record.

# Returns

A `NamedTuple` with `lower`, `upper`, `relaxed` (a `Bool`) and `note`.

# Details

If frozen positions inside a group already exceed its upper bound, the
constraint is infeasible before the optimiser starts. The choices are to fail,
to drop the constraint, or to widen it to exactly the frozen total. The third
is right, and the audit record is what makes it acceptable.

# Notes

  - **Never widen silently.** The returned `note` exists to be logged and
    surfaced. A constraint that quietly stops binding is worse than one that
    fails loudly, because the report still claims it was enforced.
  - The bound is widened to *exactly* the frozen total, never further, so the
    tradable names get no extra room.
"""
function relax_group_bound(w_frozen_total::Real, lower::Real, upper::Real;
                           group_name::AbstractString = "group")
    if w_frozen_total > upper
        return (; lower = float(lower), upper = float(w_frozen_total), relaxed = true,
                note = "$(group_name): upper bound raised from $(upper) to $(w_frozen_total) because frozen holdings already exceed it")
    end
    return (; lower = float(lower), upper = float(upper), relaxed = false, note = "")
end

end # module PortfolioState
