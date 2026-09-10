"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the rules that decide what an available-case fit does with a delisted asset.

All concrete subtypes should subtype `AbstractCoverageAlgorithm`. The family is the `alg` field of a [`CoveragePolicy`](@ref), and it exists because a delisting has no single right answer: one caller keeps the history until the asset leaves the frame, another throws it away so that a relisting starts cold, and a third holds the asset in the frame for a stated number of observations after it goes.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractCoverageAlgorithm` and implement the following methods:

## fold_inactive!

  - `fold_inactive!(alg::AbstractCoverageAlgorithm, state::AbstractPartialFitState, ni::AbstractVector{<:Bool}) -> AbstractPartialFitState`: The fold-time hook, called once per observation with the assets that have just become inactive.

### Arguments

  - `alg`: The concrete subtype instance.
  - `state`: The partial-fit state being folded, mutated in place.
  - `ni`: One entry per asset, `true` where the asset was active at the previous observation and is inactive at this one.

### Returns

  - `state::AbstractPartialFitState`: The state to fold the observation into.

## admits

  - `admits(alg::AbstractCoverageAlgorithm, share::Real, active::Bool, stale::Integer, min_coverage::Real) -> Bool`: The read-out predicate, called once per asset.

### Arguments

  - `alg`: The concrete subtype instance.
  - `share`: The asset's coverage share, its own observation count over the number of observations folded.
  - `active`: Whether the asset was active at the last observation.
  - `stale`: The number of observations folded since the asset was last finite and active.
  - `min_coverage`: The coverage floor of the policy.

### Returns

  - `admitted::Bool`: Whether the asset appears in the answer, rather than as `NaN`.

### Examples

```jldoctest
julia> struct KeepEverything <: PortfolioOptimisers.AbstractCoverageAlgorithm end

julia> PortfolioOptimisers.admits(::KeepEverything, args...) = true;

julia> PortfolioOptimisers.admits(KeepEverything(), 0.0, false, 99, 1.0)
true
```

# Related

  - [`DecayCoverage`](@ref)
  - [`ResetCoverage`](@ref)
  - [`ExpireCoverage`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`fold_inactive!`](@ref)
  - [`admits`](@ref)
"""
abstract type AbstractCoverageAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Keeps a delisted asset's history, and drops the asset from the frame the moment it goes inactive.

This is the default rule, and the cheapest of the three, because it does nothing at fold time. It is also the one whose relisting resumes: an asset that lists, delists and lists again folds into the counts and accumulators it left behind rather than starting from zero.

# Related

  - [`AbstractCoverageAlgorithm`](@ref)
  - [`ResetCoverage`](@ref)
  - [`ExpireCoverage`](@ref)
  - [`CoveragePolicy`](@ref)
"""
struct DecayCoverage <: AbstractCoverageAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Throws a delisted asset's history away, so that a relisting starts the asset cold.

The reset zeroes the asset's counts and the row and column of every accumulator that touches it, which is what [`ExpWeightedCovariance`](@ref) already does for the exponentially weighted family. Use it when a relisted ticker is a different company, or when a corporate action makes the old series incomparable with the new one.

# Related

  - [`AbstractCoverageAlgorithm`](@ref)
  - [`DecayCoverage`](@ref)
  - [`ExpireCoverage`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`fold_inactive!`](@ref)
"""
struct ResetCoverage <: AbstractCoverageAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Holds a delisted asset in the frame for a stated number of observations, then drops it.

The count is staleness, the number of observations folded since the asset was last finite and active, so a holiday spends it as freely as a delisting does and an asset that quotes again resets it to zero. `after = 0` admits exactly what [`DecayCoverage`](@ref) admits: an asset that is inactive at an observation has its staleness raised by that observation, so no state a fold produces carries an inactive asset at staleness zero.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpireCoverage(; after::Integer = 0) -> ExpireCoverage

Keywords correspond to the struct's fields.

## Validation

  - `after >= 0`.

# Examples

```jldoctest
julia> ExpireCoverage(; after = 21)
ExpireCoverage
  after ┴ Int64: 21
```

# Related

  - [`AbstractCoverageAlgorithm`](@ref)
  - [`DecayCoverage`](@ref)
  - [`ResetCoverage`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`admits`](@ref)
"""
@concrete struct ExpireCoverage <: AbstractCoverageAlgorithm
    """
    `after`: Number of observations an inactive asset stays in the frame for, counted from the last observation at which it was finite and active.
    """
    after
    function ExpireCoverage(after::Integer)
        assert_nonneg(after, :after)
        return new{typeof(after)}(after)
    end
end
function ExpireCoverage(; after::Integer = 0)::ExpireCoverage
    return ExpireCoverage(after)
end
"""
$(DocStringExtensions.TYPEDEF)

Fits each cell of a moment on the observations that cell has, instead of on the Coverage Universe.

The opt-in that replaces all-or-nothing coverage by available-case estimation. It is the `cvg` field of a moment estimator, it is `nothing` there by default, and the `nothing` arm is today's reduce-and-expand path read by dispatch, so a caller who asks for nothing pays nothing. With a policy set, every cell of the answer is fitted on the observations at which every asset of that cell is finite and active, each cell carries its own denominator, and an asset reaches the answer only where [`admits`](@ref) says so.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CoveragePolicy(;
        min_coverage::Real = 0.0,
        alg::AbstractCoverageAlgorithm = DecayCoverage()
    ) -> CoveragePolicy

Keywords correspond to the struct's fields.

## Validation

  - `0 <= min_coverage <= 1`.

# Examples

```jldoctest
julia> CoveragePolicy(; min_coverage = 0.25)
CoveragePolicy
  min_coverage ┼ Float64: 0.25
           alg ┴ DecayCoverage()
```

# Related

  - [`AbstractCoverageAlgorithm`](@ref)
  - [`DecayCoverage`](@ref)
  - [`ResetCoverage`](@ref)
  - [`ExpireCoverage`](@ref)
  - [`admits`](@ref)
  - [`CoverageCounts`](@ref)
"""
@concrete struct CoveragePolicy <: AbstractEstimator
    """
    `min_coverage`: Coverage floor, a share in `[0, 1]`. An asset whose own observation count over the number of observations folded falls below it is `NaN` in the answer.
    """
    min_coverage
    """
    `alg`: Coverage algorithm, the rule that decides what happens to a delisted asset.
    """
    alg
    function CoveragePolicy(min_coverage::Real, alg::AbstractCoverageAlgorithm)
        @argcheck(zero(min_coverage) <= min_coverage <= one(min_coverage),
                  DomainError(min_coverage,
                              "`min_coverage` is a share of the observations folded, so it must lie in [0, 1]."))
        return new{typeof(min_coverage), typeof(alg)}(min_coverage, alg)
    end
end
function CoveragePolicy(; min_coverage::Real = 0.0,
                        alg::AbstractCoverageAlgorithm = DecayCoverage())::CoveragePolicy
    return CoveragePolicy(min_coverage, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Carries the per-cell denominators and the per-asset bookkeeping of an available-case fold.

The component a partial-fit state holds in its `cvg` field, which is `nothing` on the plain path so that the plain state costs nothing. `nu` has the shape of the accumulator it serves, because the observation count of a covariance cell is a count per pair and not a count per asset, and `centre` has that shape wherever a cell's own centre differs from the state's per-asset mean.

This type is an implementation detail and is not intended for direct use. [`partial_fit!`](@ref) writes it, the read-out verbs divide by it, and [`merge_states`](@ref) folds two of them.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`CoveragePolicy`](@ref)
  - [`coverage_counts_seed`](@ref)
  - [`coverage_admission`](@ref)
  - [`partial_fit!`](@ref)
"""
@concrete struct CoverageCounts
    """
    `nu`: Per-cell surviving observation count, of the shape of the accumulator it serves.
    """
    nu
    """
    `centre`: Per-cell running mean, of the shape of the accumulator it serves, or `nothing` when the state's own `mu` is already the cell's centre. Entry `(i, j)` is the running mean of asset `i` over the observations at which the pair `(i, j)` is valid, so entry `(j, i)` is the mean of asset `j` over those same observations.
    """
    centre
    """
    `active`: Active mask of the last observation folded, one entry per asset.
    """
    active
    """
    `stale`: Number of observations folded since each asset was last finite and active, one entry per asset.
    """
    stale
end
"""
    coverage_counts_seed(cvg::Nothing, counts, N::Integer, Tf::Type,
                         pairwise::Bool) -> Nothing
    coverage_counts_seed(cvg::CoveragePolicy, counts::CoverageCounts, N::Integer, Tf::Type,
                         pairwise::Bool) -> CoverageCounts
    coverage_counts_seed(cvg::CoveragePolicy, counts::Nothing, N::Integer, Tf::Type,
                         pairwise::Bool) -> CoverageCounts

Returns the [`CoverageCounts`](@ref) an available-case fold writes into, seeding one of zeros when the state carries none.

The seed is written here rather than inside [`partial_fit!`](@ref), so that the branch that reads the `cvg` field of a state has one home and the fold reads as one line. The policy is the first argument because it is what selects the arm: `nothing` is the plain path and gives `nothing`, whatever the state carries.

# Arguments

  - `cvg`: The policy the estimator carries.
  - `counts`: The component the state carries, or `nothing`.
  - `N`: Number of assets.
  - `Tf`: Element type of the accumulators.
  - `pairwise`: Whether the cell axis is a pair, so that `nu` and `centre` are `N × N` rather than `N`-long.

# Returns

  - `counts::Option{<:CoverageCounts}`: The component `counts` holds, or a fresh one of zeros.

# Related

  - [`CoverageCounts`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`partial_fit!`](@ref)
"""
function coverage_counts_seed(::Nothing, ::Option{<:CoverageCounts}, ::Integer, ::Type,
                              ::Bool)
    return nothing
end
function coverage_counts_seed(::CoveragePolicy, counts::CoverageCounts, ::Integer, ::Type,
                              ::Bool)
    return counts
end
function coverage_counts_seed(::CoveragePolicy, ::Nothing, N::Integer, Tf::Type,
                              pairwise::Bool)
    return if pairwise
        CoverageCounts(zeros(Int, N, N), zeros(Tf, N, N), trues(N), zeros(Int, N))
    else
        CoverageCounts(zeros(Int, N), nothing, trues(N), zeros(Int, N))
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`CoverageCounts`](@ref), so that the copy shares no array with the original.

# Arguments

  - `x`: The component to copy.

# Returns

  - `counts::CoverageCounts`: A fresh component, equal to `x`, whose arrays are fresh.

# Related

  - [`CoverageCounts`](@ref)
  - [`partial_fit`](@ref)
"""
function Base.copy(x::CoverageCounts)
    return CoverageCounts(copy(x.nu), isnothing(x.centre) ? nothing : copy(x.centre),
                          copy(x.active), copy(x.stale))
end
"""
    coverage_counts_view(x::Nothing, i) -> Nothing
    coverage_counts_view(x::CoverageCounts, i) -> CoverageCounts

Slices a [`CoverageCounts`](@ref) to the selected assets.

A per-cell count reads the assets of its own cell alone, so the slice of the component is the component of the sliced universe, entry for entry. The slice copies by index and does not `view`, as [`port_opt_view`](@ref) does for every partial-fit state.

# Arguments

  - `x`: The component to slice, or `nothing`.
  - `i`: Index or indices of the assets to keep.

# Returns

  - `counts::Option{<:CoverageCounts}`: The component of the same sample over the selected assets.

# Related

  - [`CoverageCounts`](@ref)
  - [`port_opt_view`](@ref)
"""
function coverage_counts_view(::Nothing, i)
    return nothing
end
function coverage_counts_view(x::CoverageCounts, i)
    nu, centre = if isa(x.nu, AbstractMatrix)
        x.nu[i, i], x.centre[i, i]
    else
        x.nu[i], nothing
    end
    return CoverageCounts(nu, centre, x.active[i], x.stale[i])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the assets that are valid at one observation, and the assets that have just gone inactive.

An asset is **valid** when its return is finite and the active mask admits it, which is the condition the exponentially weighted family already reads. With no mask every asset is active, so a non-finite return reads as a holiday rather than as a delisting and no asset is ever newly inactive.

# Arguments

  - `x`: One observation, one entry per asset.
  - `active_mask`: The active mask of the Asset Panel at this observation, or `nothing`.
  - `counts`: The component whose `active` field holds the mask of the previous observation.

# Validation

  - `active_mask`, when it is given, has one entry per asset. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `(valid, ni)::Tuple`: The valid assets, and the assets that were active at the previous observation and are inactive at this one.

# Related

  - [`CoverageCounts`](@ref)
  - [`fold_inactive!`](@ref)
  - [`partial_fit!`](@ref)
"""
function coverage_valid(x::VecNum, active_mask::Option{<:AbstractVector{<:Bool}},
                        counts::CoverageCounts)
    finite = isfinite.(x)
    return if isnothing(active_mask)
        finite, falses(length(x))
    else
        @argcheck(length(active_mask) == length(x),
                  DimensionMismatch("the active mask must have one entry per asset, but the observation has $(length(x)) entries and the mask has $(length(active_mask))."))
        finite .& active_mask, (.!active_mask) .& counts.active
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Moves the per-asset bookkeeping of a [`CoverageCounts`](@ref) on by one observation.

# Algorithm

 1. Set the staleness of every valid asset to zero, and add one to the staleness of every other.
 2. Rebind `active` to the mask of this observation, which is every asset when the caller gave none.

# Arguments

  - `counts`: The component to move on, mutated in place.
  - `valid`: The valid assets of this observation.
  - `active_mask`: The active mask of this observation, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`CoverageCounts`](@ref)
  - [`coverage_valid`](@ref)
"""
function coverage_step!(counts::CoverageCounts, valid::AbstractVector{<:Bool},
                        active_mask::Option{<:AbstractVector{<:Bool}})::Nothing
    for i in eachindex(valid)
        counts.stale[i] = valid[i] ? 0 : counts.stale[i] + 1
    end
    if isnothing(active_mask)
        counts.active .= true
    else
        counts.active .= active_mask
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the staleness counters of two [`CoverageCounts`](@ref) fitted on consecutive blocks.

An asset that was valid somewhere in the second block carries that block's own count, and an asset that was valid nowhere in it has spent the whole block stale, so its staleness is the two counts added. The test for "valid nowhere in `b`" is that `b`'s counter has reached `b`'s observation count, which is what the counter does when nothing resets it.

# Arguments

  - `a`: The component of the first block of observations.
  - `b`: The component of the second block of observations.
  - `n_b`: The number of observations folded into `b`.

# Returns

  - `stale::Vector{<:Integer}`: The staleness of each asset over the concatenated block.

# Related

  - [`CoverageCounts`](@ref)
  - [`merge_states`](@ref)
  - [`coverage_step!`](@ref)
"""
function coverage_merge_stale(a::CoverageCounts, b::CoverageCounts, n_b::Integer)
    stale = similar(b.stale)
    for i in eachindex(stale, a.stale, b.stale)
        stale[i] = b.stale[i] == n_b ? a.stale[i] + b.stale[i] : b.stale[i]
    end
    return stale
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads everything a two-pass available-case fit needs out of a block of observations.

The block counterpart of [`coverage_valid`](@ref) and [`coverage_step!`](@ref), for the arms that have no incremental recursion and must see the whole window at once. It returns the oriented block beside the four quantities the fold would otherwise have carried, so that a two-pass arm and a folding arm read the same universe and the same bookkeeping.

# Algorithm

 1. Orient `X`, and the mask when there is one, to `observations × assets`.
 2. Take the valid entries, finite and active, giving `F`.
 3. Take each asset's available-case mean over its own valid entries, giving `mu`, which is zero for an asset with none.
 4. Take the active mask of the last observation, and each asset's staleness at it.

# Arguments

  - `X`: The block of observations.
  - `active_mask`: The active mask of the Asset Panel over the block, or `nothing`.
  - `dims`: Whether the observations lie on the rows, `1`, or on the columns, `2`.

# Validation

  - `active_mask`, when it is given, has the shape of `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `(Xo, F, mu, active, stale)::Tuple`: The oriented block, its valid entries, each asset's available-case mean, the active mask of the last observation, and each asset's staleness at it.

# Related

  - [`coverage_valid`](@ref)
  - [`coverage_step!`](@ref)
  - [`CoverageCounts`](@ref)
"""
function coverage_valid_block(X::MatNum, active_mask::Option{<:AbstractMatrix{<:Bool}};
                              dims::Int = 1)
    Xo = dims_oriented(dims, X)
    F = isfinite.(Xo)
    if !isnothing(active_mask)
        amsk = dims_oriented(dims, active_mask)
        @argcheck(size(amsk) == size(Xo),
                  DimensionMismatch("size(X) ($(size(Xo))) must match size(active_mask) ($(size(amsk)))"))
        F .&= amsk
    end
    T, N = size(Xo)
    Tf = float(eltype(Xo))
    mu = zeros(Tf, N)
    stale = zeros(Int, N)
    for j in axes(Xo, 2)
        s = zero(Tf)
        c = 0
        for t in axes(Xo, 1)
            if F[t, j]
                s += Xo[t, j]
                c += 1
            end
        end
        mu[j] = iszero(c) ? zero(Tf) : s / c
        k = findlast(view(F, :, j))
        stale[j] = isnothing(k) ? T : T - k
    end
    active = if isnothing(active_mask)
        trues(N)
    else
        BitVector(view(dims_oriented(dims, active_mask), T, :))
    end
    return Xo, F, mu, active, stale
end
"""
    fold_inactive!(alg, state, ni)

Applies a coverage algorithm's fold-time rule to the assets that have just become inactive.

The first of the two verbs of the [`AbstractCoverageAlgorithm`](@ref) interface. It runs once per observation, before the observation is folded, so a rule that throws history away runs while the state still holds the history to throw.

# Arguments

  - `alg`: The coverage algorithm.
  - `state`: The partial-fit state, mutated in place.
  - `ni`: One entry per asset, `true` where the asset was active at the previous observation and is inactive at this one.

# Returns

  - `state`: The state to fold the observation into.

# Related

  - [`AbstractCoverageAlgorithm`](@ref)
  - [`ResetCoverage`](@ref)
  - [`admits`](@ref)
  - [`partial_fit!`](@ref)
"""
function fold_inactive! end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`DecayCoverage`](@ref) and [`ExpireCoverage`](@ref) method of [`fold_inactive!`](@ref). Both keep the history of a delisted asset, so neither has anything to do at fold time and the state passes through.

# Related

  - [`fold_inactive!`](@ref)
  - [`DecayCoverage`](@ref)
  - [`ExpireCoverage`](@ref)
"""
function fold_inactive!(::Union{<:DecayCoverage, <:ExpireCoverage},
                        state::AbstractPartialFitState, ::AbstractVector{<:Bool})
    return state
end
"""
    coverage_reset!(A::Nothing, ni::AbstractVector{<:Bool}) -> Nothing
    coverage_reset!(A::AbstractVector, ni::AbstractVector{<:Bool}) -> Nothing
    coverage_reset!(A::AbstractMatrix, ni::AbstractVector{<:Bool}) -> Nothing

Zeroes the entries of an accumulator that touch a set of assets.

The primitive the [`ResetCoverage`](@ref) methods of [`fold_inactive!`](@ref) share, so that the reset of a per-asset accumulator and the reset of a per-pair accumulator are written once.

# Arguments

  - `A`: The accumulator or count array to zero, mutated in place. `nothing` passes through, which is the `centre` of a per-asset state.
  - `ni`: One entry per asset, `true` where the asset is to be zeroed.

# Returns

  - `nothing`.

# Related

  - [`ResetCoverage`](@ref)
  - [`fold_inactive!`](@ref)
"""
function coverage_reset!(::Nothing, ::AbstractVector{<:Bool})::Nothing
    return nothing
end
function coverage_reset!(A::AbstractVector, ni::AbstractVector{<:Bool})::Nothing
    A[ni] .= zero(eltype(A))
    return nothing
end
function coverage_reset!(A::AbstractMatrix, ni::AbstractVector{<:Bool})::Nothing
    A[ni, :] .= zero(eltype(A))
    A[:, ni] .= zero(eltype(A))
    return nothing
end
"""
    admits(alg, share, active, stale, min_coverage) -> Bool

Decides whether an asset reaches the answer of an available-case fit.

The second of the two verbs of the [`AbstractCoverageAlgorithm`](@ref) interface. It runs once per asset at read-out, and the assets it refuses are the `NaN` frame the read-out writes.

# Arguments

  - `alg`: The coverage algorithm.
  - `share`: The asset's coverage share, its own observation count over the number of observations folded.
  - `active`: Whether the asset was active at the last observation.
  - `stale`: The number of observations folded since the asset was last finite and active.
  - `min_coverage`: The coverage floor of the [`CoveragePolicy`](@ref).

# Returns

  - `admitted::Bool`: Whether the asset reaches the answer.

# Related

  - [`AbstractCoverageAlgorithm`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`fold_inactive!`](@ref)
  - [`coverage_admission`](@ref)
"""
function admits end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`DecayCoverage`](@ref) and [`ResetCoverage`](@ref) method of [`admits`](@ref). An asset is admitted while it is active and its coverage share reaches the floor, so the exit is immediate under both and the history is what the two algorithms differ over.

# Related

  - [`admits`](@ref)
  - [`DecayCoverage`](@ref)
  - [`ResetCoverage`](@ref)
"""
function admits(::Union{<:DecayCoverage, <:ResetCoverage}, share::Real, active::Bool,
                ::Integer, min_coverage::Real)
    return active && share >= min_coverage
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`ExpireCoverage`](@ref) method of [`admits`](@ref). An inactive asset stays in the frame while its staleness is at most `after`, and its coverage share must reach the floor either way.

# Related

  - [`admits`](@ref)
  - [`ExpireCoverage`](@ref)
"""
function admits(alg::ExpireCoverage, share::Real, active::Bool, stale::Integer,
                min_coverage::Real)
    return share >= min_coverage && (active || stale <= alg.after)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the universe of an available-case fit out of its per-cell counts.

The read-out's own mask, and the one place [`admits`](@ref) is called. It collapses onto the `nothing` sentinel exactly as [`coverage_sentinel`](@ref) does, so an answer that admits every asset allocates no mask. Unlike [`coverage_sentinel`](@ref) it refuses no sample: an available-case fit whose window admits nothing answers all `NaN` rather than raising, because the fit is opt-in and its caller asked for the gaps.

# Algorithm

 1. Take the per-asset observation count, which is the diagonal when the counts are per pair.
 2. Divide it by `n`, the number of observations folded, giving each asset's coverage share.
 3. Call [`admits`](@ref) once per asset.
 4. Return `nothing` when every asset is admitted, and the mask otherwise.

# Arguments

  - `cvg`: The policy the estimator carries.
  - `counts`: The per-cell counts the state carries.
  - `n`: The number of observations folded, which is the share's denominator.

# Returns

  - `cmsk::Option{BitVector}`: The admitted assets, or `nothing` when every asset is admitted.

# Related

  - [`admits`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`CoverageCounts`](@ref)
  - [`coverage_divide`](@ref)
"""
function coverage_admission(cvg::CoveragePolicy, counts::CoverageCounts, n::Integer)
    nu = isa(counts.nu, AbstractMatrix) ? LinearAlgebra.diag(counts.nu) : counts.nu
    cmsk = BitVector(undef, length(nu))
    d = max(n, one(n))
    for i in eachindex(nu, cmsk)
        cmsk[i] = admits(cvg.alg, nu[i] / d, counts.active[i], counts.stale[i],
                         cvg.min_coverage)
    end
    return all(cmsk) ? nothing : cmsk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Divides an available-case accumulator by its per-cell denominator, and frames the assets the policy refuses.

The tail every available-case read-out shares. A cell whose denominator has not reached one is `NaN`, which is the per-cell half of the rule, and an asset [`coverage_admission`](@ref) refuses is `NaN` across its whole row and column, which is the per-asset half.

# Algorithm

 1. Divide each entry of `M` by its entry of `nu` less `corrected`, writing `NaN` where that denominator is below one.
 2. Write `NaN` into every entry that touches an asset outside `cmsk`.

# Arguments

  - `M`: The accumulator, of the shape of the answer.
  - `nu`: The per-cell denominator, of the shape of `M`.
  - `corrected`: The Bessel correction, subtracted from every denominator.
  - `cmsk`: The admitted assets, or `nothing`.

# Returns

  - `val::Array{<:AbstractFloat}`: The answer, of the shape of `M`, carrying `NaN` where the fit has no number.

# Related

  - [`coverage_admission`](@ref)
  - [`CoverageCounts`](@ref)
  - [`expand_moment`](@ref)
"""
function coverage_divide(M::AbstractArray, nu::AbstractArray, corrected::Bool,
                         cmsk::Option{BitVector})
    Tf = float(eltype(M))
    val = Array{Tf}(undef, size(M))
    for k in eachindex(val, M, nu)
        d = nu[k] - corrected
        val[k] = d >= one(d) ? Tf(M[k]) / d : Tf(NaN)
    end
    coverage_refuse!(val, cmsk)
    return val
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Frames an available-case estimate that needs no division, and frames the assets the policy refuses.

The read-out of a quantity that is already a ratio, which is what a running per-asset mean is: the value is copied where its cell has an observation and is `NaN` where it has none, and [`coverage_refuse!`](@ref) then writes the frame. It exists beside [`coverage_divide`](@ref) so that a Welford mean is never multiplied by its count and divided by it again, which would move the last bits of an answer the fold computed exactly.

# Arguments

  - `A`: The estimate, of the shape of the answer.
  - `nu`: The per-cell observation count, of the shape of `A`.
  - `cmsk`: The admitted assets, or `nothing`.

# Returns

  - `val::Array{<:AbstractFloat}`: The answer, of the shape of `A`, carrying `NaN` where the fit has no number.

# Related

  - [`coverage_divide`](@ref)
  - [`coverage_refuse!`](@ref)
  - [`coverage_admission`](@ref)
"""
function coverage_frame(A::AbstractArray, nu::AbstractArray, cmsk::Option{BitVector})
    Tf = float(eltype(A))
    val = Array{Tf}(undef, size(A))
    for k in eachindex(val, A, nu)
        val[k] = nu[k] >= one(nu[k]) ? Tf(A[k]) : Tf(NaN)
    end
    coverage_refuse!(val, cmsk)
    return val
end
"""
    coverage_refuse!(val::AbstractArray, cmsk::Nothing) -> Nothing
    coverage_refuse!(val::AbstractArray, cmsk::BitVector) -> Nothing

Writes `NaN` into every entry of an answer that touches an asset the policy refuses.

The per-asset half of the read-out rule, shared by [`coverage_divide`](@ref) and [`coverage_frame`](@ref). A `nothing` mask is the sentinel of [`coverage_admission`](@ref) and means that every asset is admitted, so nothing is written.

# Arguments

  - `val`: The answer, mutated in place.
  - `cmsk`: The admitted assets, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`coverage_admission`](@ref)
  - [`coverage_divide`](@ref)
  - [`coverage_frame`](@ref)
"""
function coverage_refuse!(::AbstractArray, ::Nothing)::Nothing
    return nothing
end
function coverage_refuse!(val::AbstractArray, cmsk::BitVector)::Nothing
    out = .!cmsk
    if isa(val, AbstractMatrix)
        val[out, :] .= convert(eltype(val), NaN)
        val[:, out] .= convert(eltype(val), NaN)
    else
        val[out] .= convert(eltype(val), NaN)
    end
    return nothing
end
"""
    coverage_refuse_comoment!(val::AbstractMatrix, cmsk::Nothing, ::Val) -> Nothing
    coverage_refuse_comoment!(val::AbstractMatrix, cmsk::BitVector, ::Val{:sk}) -> Nothing
    coverage_refuse_comoment!(val::AbstractMatrix, cmsk::BitVector, ::Val{:kt}) -> Nothing

Writes `NaN` into every entry of a higher-order co-moment answer that touches an asset the policy refuses.

The co-moment form of [`coverage_refuse!`](@ref), for an answer whose axes are not all the asset axis. A coskewness tensor is `assets × assets²`, so its rows take the asset mask and its columns the pair mask; a cokurtosis matrix is `assets² × assets²`, so both of its axes take the pair mask. The `Val` marker names the shape, exactly as it does for [`expand_moment`](@ref), and the mask stays one argument so the arm is a plain dispatch rather than a product of two `Option`s.

A pair is admitted when both of its assets are, and a co-moment tensor indexes the pair `(i, j)` at the column `(i - 1) * N + j`, which is the layout [`coverage_pair_index`](@ref) states. `kron(cmsk, cmsk)` is that conjunction in that order, entry for entry.

# Arguments

  - `val`: The answer, mutated in place.
  - `cmsk`: The admitted assets, or `nothing` when every asset is admitted.
  - `::Val`: `Val(:sk)` for a coskewness tensor and `Val(:kt)` for a cokurtosis matrix.

# Returns

  - `nothing`.

# Related

  - [`coverage_refuse!`](@ref)
  - [`coverage_admission`](@ref)
  - [`coverage_pair_index`](@ref)
  - [`expand_moment`](@ref)
"""
function coverage_refuse_comoment!(::AbstractMatrix, ::Nothing, ::Val)::Nothing
    return nothing
end
function coverage_refuse_comoment!(val::AbstractMatrix, cmsk::BitVector,
                                   ::Val{:sk})::Nothing
    nan = convert(eltype(val), NaN)
    val[.!cmsk, :] .= nan
    val[:, .!BitVector(kron(cmsk, cmsk))] .= nan
    return nothing
end
function coverage_refuse_comoment!(val::AbstractMatrix, cmsk::BitVector,
                                   ::Val{:kt})::Nothing
    nan = convert(eltype(val), NaN)
    pout = .!BitVector(kron(cmsk, cmsk))
    val[pout, :] .= nan
    val[:, pout] .= nan
    return nothing
end
export CoveragePolicy, DecayCoverage, ResetCoverage, ExpireCoverage
