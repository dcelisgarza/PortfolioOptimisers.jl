"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype of the rules that decide what an available-case fit does with a delisted asset.

All concrete subtypes should subtype `AbstractCoverageAlgorithm`. A member of the family goes in the `alg` field of a [`CoveragePolicy`](@ref). The family exists because a delisting has more than one correct treatment. One caller keeps the history of the asset and drops the asset at once. Another caller throws the history away, so that a relisting starts cold. A third holds the asset in the frame for a stated number of observations after it goes.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractCoverageAlgorithm` and implement the following methods:

## fold_inactive!

  - `fold_inactive!(alg::AbstractCoverageAlgorithm, state::AbstractPartialFitState, ni::AbstractVector{<:Bool}) -> AbstractPartialFitState`: The fold-time hook, called once per observation with the assets that have just become inactive. Only the arms that fold call it. The two-pass arms, which are the semi-covariance, the coskewness and the cokurtosis, read the window whole and never call it, so a new subtype keeps the history in those arms.

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

This is the default rule. It does nothing at fold time, and it does not read the staleness at read-out. A relisting resumes the history. An asset that lists, delists and lists again folds into the counts and accumulators that it left behind, and does not start from zero. [`ExpireCoverage`](@ref) keeps the history in the same way, and it differs only in how long a delisted asset stays in the frame.

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

The reset zeroes the counts of the asset, and the row and the column of every accumulator that touches it. [`ExpWeightedCovariance`](@ref) resets an inactive asset in the same way. Use it when a relisted ticker is a different company, or when a corporate action makes the old series incomparable with the new one.

Every arm applies the reset. A folding arm applies it through [`fold_inactive!`](@ref) at the observation where the asset goes inactive, and a two-pass arm applies it to the whole window through [`coverage_inactive_block!`](@ref). The reset needs an active mask. With no mask no asset goes inactive, a gap reads as a holiday, and the answer is the answer of [`DecayCoverage`](@ref). The reset does not zero the number of observations folded, so the coverage share of a relisted asset is its count since the relisting over every observation folded.

# Related

  - [`AbstractCoverageAlgorithm`](@ref)
  - [`DecayCoverage`](@ref)
  - [`ExpireCoverage`](@ref)
  - [`CoveragePolicy`](@ref)
  - [`fold_inactive!`](@ref)
  - [`coverage_inactive_block!`](@ref)
"""
struct ResetCoverage <: AbstractCoverageAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Holds a delisted asset in the frame for a stated number of observations, then drops it.

The count is the staleness of the asset, the number of observations folded since the asset was last finite and active. A holiday raises the staleness as a delisting does, so a holiday just before a delisting shortens the hold. A finite return at an active observation sets the staleness back to zero. `after = 0` admits exactly what [`DecayCoverage`](@ref) admits. An observation at which an asset is inactive raises its staleness, so no fold leaves an inactive asset at a staleness of zero.

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

A policy replaces the all-or-nothing Coverage Universe with available-case estimation. It goes in the `cvg` field of a moment estimator, where the default is `nothing`. With `nothing` the estimator takes the reduce-and-expand path by dispatch, and the policy costs nothing. With a policy set, the estimator fits every cell of the answer on the observations at which every asset of that cell is finite and active, and each cell carries its own denominator. An asset reaches the answer only where [`admits`](@ref) lets it in.

**`min_coverage` is also the share that a [`scenario_fill`](@ref) passes in silence.** [`admits`](@ref) reads the coverage share of an asset as its own observation count over the number of observations folded. The fill counts the non-finite entries of the column over the same denominator. So a column that [`DecayCoverage`](@ref), [`ResetCoverage`](@ref) or [`ExpireCoverage`](@ref) admits is at most a `1 - min_coverage` share non-finite. Where the `fill_limit` of a prior is `nothing`, [`resolve_fill_limit`](@ref) derives `1 - min_coverage`, and no admitted column goes past it. A policy therefore turns on no warning, and an available-case walk-forward gives no warning for the gaps that it was set up to fill.

The default `min_coverage = 0` admits every column, and a prior fills in silence every column that has a variance. Under the default Bessel correction that is a column with two or more observations. A column with one observation has a `NaN` variance, so the Investable Mask drops the asset and nothing is filled. To admit broadly and still get the warning, set the `fill_limit` of the prior explicitly, tighter than `1 - min_coverage`.

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
    `min_coverage`: Coverage floor, a share in `[0, 1]`. An asset whose own observation count over the number of observations folded falls below it is `NaN` in the answer. It is also the share that a [`scenario_fill`](@ref) fills in silence, so under the default of `0` a prior fills every column that has a variance and says nothing.
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

A partial-fit state holds it in its `cvg` field. On the plain path that field is `nothing`, so the plain state costs nothing. `nu` has the shape of the accumulator it serves, because the observation count of a covariance cell is a count per pair and not a count per asset. `centre` has that shape too wherever the centre of a cell differs from the per-asset mean of the state.

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

[`partial_fit!`](@ref) calls it, so one function holds the branch on the `cvg` field of a state. The policy is the first argument because it selects the arm. `nothing` is the plain path and gives `nothing`, whatever the state carries.

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

An asset is **valid** when its return is finite and the active mask admits it. The exponentially weighted family reads the same condition. With no mask every asset is active, so a non-finite return reads as a holiday and not as a delisting, and no asset ever goes inactive.

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
 2. Overwrite `active` in place with the mask of this observation, or with `true` for every asset when the caller gave no mask.

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

An asset that was valid somewhere in the second block takes the count of that block. An asset that was valid nowhere in it was stale for the whole block, so its staleness is the sum of the two counts. The counter of `b` reaches the observation count of `b` exactly when the asset was valid nowhere in `b`, because only a valid observation sets the counter back to zero.

# Mathematical definition

```math
\\begin{align}
\\tau_{i} &= \\begin{cases}
\\tau_{i}^{a} + \\tau_{i}^{b} & \\text{if } \\tau_{i}^{b} = n_{b}\\,, \\\\
\\tau_{i}^{b} & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:tau_i_cvg])
  - ``\\tau_{i}^{a}``, ``\\tau_{i}^{b}``: Staleness of asset ``i`` in `a` and in `b`.
  - ``n_{b}``: Number of observations folded into `b`.

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

It is the block counterpart of [`coverage_valid`](@ref), [`fold_inactive!`](@ref) and [`coverage_step!`](@ref), for the arms that have no incremental recursion and must see the whole window at once. It returns the oriented block beside the four quantities that a fold carries, so that a two-pass arm and a folding arm read the same universe and the same bookkeeping.

# Algorithm

 1. Orient `X`, and the mask when there is one, to `observations × assets`.
 2. Take the valid entries, finite and active, giving `F`.
 3. Take the active mask of the last observation, giving `active`, and the number of observations after the last valid entry of each asset, giving `stale`. An asset with no valid entry has a staleness of `T`, the number of observations.
 4. Apply the fold-time rule of `alg` to `F` with [`coverage_inactive_block!`](@ref).
 5. Take the available-case mean of each asset over its own entries of `F`, giving `mu`. It is zero for an asset with none.

# Arguments

  - `X`: The block of observations.
  - `active_mask`: The active mask of the Asset Panel over the block, or `nothing`.
  - `alg`: The coverage algorithm of the policy.
  - `dims`: Whether the observations lie on the rows, `1`, or on the columns, `2`.

# Validation

  - `active_mask`, when it is given, has the shape of `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `(Xo, F, mu, active, stale)::Tuple`: The oriented block, its valid entries after the fold-time rule, each asset's available-case mean, the active mask of the last observation, and each asset's staleness at it.

# Related

  - [`coverage_valid`](@ref)
  - [`coverage_step!`](@ref)
  - [`coverage_inactive_block!`](@ref)
  - [`CoverageCounts`](@ref)
"""
function coverage_valid_block(X::MatNum, active_mask::Option{<:AbstractMatrix{<:Bool}},
                              alg::AbstractCoverageAlgorithm; dims::Int = 1)
    Xo = dims_oriented(dims, X)
    F = isfinite.(Xo)
    amsk = isnothing(active_mask) ? nothing : dims_oriented(dims, active_mask)
    if !isnothing(amsk)
        @argcheck(size(amsk) == size(Xo),
                  DimensionMismatch("size(X) ($(size(Xo))) must match size(active_mask) ($(size(amsk)))"))
        F .&= amsk
    end
    T, N = size(Xo)
    stale = [T - something(findlast(view(F, :, j)), 0) for j in axes(F, 2)]
    coverage_inactive_block!(alg, F, amsk)
    Tf = typeof(zero(eltype(Xo)) / one(Int))
    mu = zeros(Tf, N)
    for j in axes(Xo, 2)
        c = count(view(F, :, j))
        s = sum(Xo[t, j] for t in axes(Xo, 1) if F[t, j]; init = zero(Tf))
        mu[j] = iszero(c) ? zero(Tf) : s / c
    end
    active = isnothing(amsk) ? trues(N) : BitVector(view(amsk, T, :))
    return Xo, F, mu, active, stale
end
"""
    coverage_inactive_block!(alg::AbstractCoverageAlgorithm, F::AbstractMatrix{Bool},
                             amsk::Option{<:AbstractMatrix{<:Bool}}) -> Nothing
    coverage_inactive_block!(alg::ResetCoverage, F::AbstractMatrix{Bool},
                             amsk::AbstractMatrix{<:Bool}) -> Nothing

Applies the fold-time rule of a coverage algorithm to the valid entries of a whole block.

It is the block form of [`fold_inactive!`](@ref), for the two-pass arms that never fold. A fold applies the rule at each observation where an asset goes inactive. A block applies the same rule to the valid entries of the window at once, so that the two arms admit the same observations.

The first method passes the block through. It serves [`DecayCoverage`](@ref) and [`ExpireCoverage`](@ref), which keep the history, and [`ResetCoverage`](@ref) with no active mask, where no asset ever goes inactive. It also serves a coverage algorithm of the caller's own, whose [`fold_inactive!`](@ref) method the two-pass arms cannot call, so such an algorithm keeps the history in those arms.

# Algorithm

The [`ResetCoverage`](@ref) method runs these steps for each asset `j`:

 1. Find the last observation `t0` at which the asset goes inactive, where the asset is inactive at `t0` and active at the observation before it. The observation before the first one counts as active, as it does for the seed of a fold.
 2. Clear the valid entries of the asset at every observation up to `t0`. An asset that never goes inactive has `t0 = 0`, and keeps every entry.

# Arguments

  - `alg`: The coverage algorithm.
  - `F`: The valid entries of the block, `observations × assets`, mutated in place.
  - `amsk`: The active mask of the block in the orientation of `F`, or `nothing`.

# Returns

  - `nothing`.

# Related

  - [`fold_inactive!`](@ref)
  - [`ResetCoverage`](@ref)
  - [`coverage_valid_block`](@ref)
"""
function coverage_inactive_block!(::AbstractCoverageAlgorithm, ::AbstractMatrix{Bool},
                                  ::Option{<:AbstractMatrix{<:Bool}})::Nothing
    return nothing
end
function coverage_inactive_block!(::ResetCoverage, F::AbstractMatrix{Bool},
                                  amsk::AbstractMatrix{<:Bool})::Nothing
    for j in axes(F, 2)
        t0 = 0
        prev = true
        for t in axes(amsk, 1)
            cur = amsk[t, j]
            if prev && !cur
                t0 = t
            end
            prev = cur
        end
        F[1:t0, j] .= false
    end
    return nothing
end
"""
    fold_inactive!(alg, state, ni)

Applies a coverage algorithm's fold-time rule to the assets that have just become inactive.

It is the first of the two verbs of the [`AbstractCoverageAlgorithm`](@ref) interface. It runs once per observation, before the fold of that observation, so a rule that throws the history away runs while the state still holds that history. Only the folding arms call it. A two-pass arm applies the rule of a built-in algorithm to the whole window through [`coverage_inactive_block!`](@ref).

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
  - [`coverage_inactive_block!`](@ref)
"""
function fold_inactive! end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`DecayCoverage`](@ref) and [`ExpireCoverage`](@ref) method of [`fold_inactive!`](@ref). Both keep the history of a delisted asset, so neither does anything at fold time, and the state passes through.

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

The [`ResetCoverage`](@ref) methods of [`fold_inactive!`](@ref) share it, so the reset of a per-asset accumulator and the reset of a per-pair accumulator are each written once.

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

It is the second of the two verbs of the [`AbstractCoverageAlgorithm`](@ref) interface. It runs once per asset at read-out, and the read-out writes `NaN` across every asset that it refuses.

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

[`DecayCoverage`](@ref) and [`ResetCoverage`](@ref) method of [`admits`](@ref). An asset is admitted while it is active and its coverage share reaches the floor. So both algorithms drop a delisted asset at once, and they differ only in the history they keep.

# Mathematical definition

```math
\\begin{align}
i \\in \\mathcal{A} &\\iff a_{i} = 1 \\land s_{i} \\geq c\\,.
\\end{align}
```

Where:

  - $(math_dict[:A_adm_cvg])
  - $(math_dict[:a_i_cvg])
  - $(math_dict[:s_i_cvg])
  - $(math_dict[:c_cvg])

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

[`ExpireCoverage`](@ref) method of [`admits`](@ref). An inactive asset stays in the frame while its staleness is at most `after`, and its coverage share must reach the floor in both cases.

# Mathematical definition

```math
\\begin{align}
i \\in \\mathcal{A} &\\iff s_{i} \\geq c \\land \\left(a_{i} = 1 \\lor \\tau_{i} \\leq h\\right)\\,.
\\end{align}
```

Where:

  - $(math_dict[:A_adm_cvg])
  - $(math_dict[:s_i_cvg])
  - $(math_dict[:c_cvg])
  - $(math_dict[:a_i_cvg])
  - $(math_dict[:tau_i_cvg])
  - ``h``: The `after` field of the algorithm.

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

It makes the mask of the read-out, and it is the one place that calls [`admits`](@ref). It collapses onto the `nothing` sentinel exactly as [`coverage_sentinel`](@ref) does, so an answer that admits every asset allocates no mask. Unlike [`coverage_sentinel`](@ref) it refuses no sample. An available-case fit whose window admits nothing answers all `NaN` and does not raise, because the fit is opt-in and its caller asked for the gaps.

# Mathematical definition

```math
\\begin{align}
s_{i} &= \\frac{\\nu_{i}}{\\max(n, 1)}\\,.
\\end{align}
```

Where:

  - $(math_dict[:s_i_cvg])
  - ``\\nu_{i}``: Observation count of asset ``i``, the diagonal entry ``\\nu_{ii}`` when the counts are per pair.
  - ``n``: Number of observations folded. A state that folded none gives every asset a share of zero.

# Algorithm

 1. Take the per-asset observation count `nu`, which is the diagonal when the counts are per pair.
 2. Take the coverage share of each asset.
 3. Call [`admits`](@ref) once per asset, giving the mask `cmsk`.
 4. Return `nothing` when every asset is admitted, and `cmsk` otherwise.

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

Every available-case read-out ends with it. A cell whose denominator has not reached one is `NaN`, which is the per-cell half of the rule. An asset that [`coverage_admission`](@ref) refuses is `NaN` across its whole row and column, which is the per-asset half.

The element type of the answer is the type of the division itself, so a `Float32` accumulator reads out as `Float32`, and the sentinel does not widen it. An element type that cannot hold `NaN` cannot hold the answer either. An exact accumulator, a `Rational` among them, raises an `InexactError` at the first cell that the policy refuses, and does not read out as `Float64`. A window with no refused cell is unaffected.

# Mathematical definition

```math
\\begin{align}
v_{k} &= \\begin{cases}
\\dfrac{M_{k}}{\\nu_{k} - \\delta} & \\text{if } \\nu_{k} - \\delta \\geq 1 \\text{ and cell } k \\text{ is admitted}\\,, \\\\
\\mathrm{NaN} & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:v_k_cvg])
  - ``M_{k}``: Entry ``k`` of the accumulator.
  - $(math_dict[:nu_k_cvg])
  - ``\\delta``: The Bessel correction, ``1`` when `corrected` is `true` and ``0`` otherwise.
  - $(math_dict[:A_adm_cvg])

# Algorithm

 1. Divide each entry of `M` by its corrected denominator, giving `val`, with `NaN` where the corrected denominator is below one.
 2. Write `NaN` into every entry of `val` that touches an asset outside `cmsk`, with [`coverage_refuse!`](@ref).

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
    Tf = typeof(zero(eltype(M)) / one(eltype(nu)))
    val = Array{Tf}(undef, size(M))
    for k in eachindex(val, M, nu)
        d = nu[k] - corrected
        val[k] = d >= one(d) ? M[k] / d : Tf(NaN)
    end
    coverage_refuse!(val, cmsk)
    return val
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Frames an available-case estimate that needs no division, and frames the assets the policy refuses.

It reads out a quantity that is already a ratio, such as a running per-asset mean. It copies the value where its cell has an observation and writes `NaN` where the cell has none, and [`coverage_refuse!`](@ref) then writes the frame. It is separate from [`coverage_divide`](@ref) so that a Welford mean is never multiplied by its count and divided by it again, which would change the last bits of the mean that the fold computed.

The answer carries the element type of `A`, which the fold derived from its own division. As in [`coverage_divide`](@ref), an exact element type cannot hold the `NaN` sentinel, and it raises an `InexactError` at the first cell that has no observation.

# Mathematical definition

```math
\\begin{align}
v_{k} &= \\begin{cases}
A_{k} & \\text{if } \\nu_{k} \\geq 1 \\text{ and cell } k \\text{ is admitted}\\,, \\\\
\\mathrm{NaN} & \\text{otherwise}\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:v_k_cvg])
  - ``A_{k}``: Entry ``k`` of the estimate.
  - $(math_dict[:nu_k_cvg])
  - $(math_dict[:A_adm_cvg])

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
    Tf = eltype(A)
    val = Array{Tf}(undef, size(A))
    for k in eachindex(val, A, nu)
        val[k] = nu[k] >= one(nu[k]) ? A[k] : Tf(NaN)
    end
    coverage_refuse!(val, cmsk)
    return val
end
"""
    coverage_refuse!(val::AbstractArray, cmsk::Nothing) -> Nothing
    coverage_refuse!(val::AbstractArray, cmsk::BitVector) -> Nothing

Writes `NaN` into every entry of an answer that touches an asset the policy refuses.

It is the per-asset half of the read-out rule, and [`coverage_divide`](@ref) and [`coverage_frame`](@ref) share it. A `nothing` mask is the sentinel of [`coverage_admission`](@ref). It means that every asset is admitted, so the method writes nothing.

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

It is the co-moment form of [`coverage_refuse!`](@ref), for an answer whose axes are not all the asset axis. A coskewness tensor is `assets × assets²`, so its rows take the asset mask and its columns take the pair mask. A cokurtosis matrix is `assets² × assets²`, so both of its axes take the pair mask. The `Val` marker names the shape, as it does for [`expand_moment`](@ref). The mask stays one argument, so the arm is a plain dispatch and not a product of two `Option`s.

A pair is admitted when both of its assets are. A co-moment tensor indexes the pair `(i, j)` at the column `(i - 1) * N + j`, which is the layout that [`coverage_pair_index`](@ref) states. `kron(cmsk, cmsk)` is that conjunction in that order, entry for entry.

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
public AbstractCoverageAlgorithm, fold_inactive!, admits
