"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all cross-sectional transform types.

A member rescales or reshapes one observation of an `observations × assets` matrix against the other assets of that same observation, so no member reads a second observation and no member is fitted. Two members treat an outlier and three members turn a raw quantity into a score.

All concrete and/or abstract types representing cross-sectional transforms should be subtypes of `AbstractCrossSectionalTransform`.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractCrossSectionalTransform` and implement the following method:

## `cross_sectional_transform`

  - [`cross_sectional_transform(ct::AbstractCrossSectionalTransform, X::MatNum; w::Option{<:MatNum} = nothing, groups::Option{<:AbstractMatrix{<:Integer}} = nothing)`](@ref): Returns the transformed matrix.

### Arguments

  - `ct`: The concrete subtype instance.
  - `X`: Data matrix `observations × assets`, where a `NaN` marks a missing cell.
  - `w`: Benchmark weight matrix `observations × assets`, or `nothing`.
  - `groups`: Group label matrix `observations × assets`, or `nothing`.

### Returns

  - `Y::Matrix{<:AbstractFloat}`: Transformed matrix `observations × assets`.

# Related

  - [`AbstractEstimator`](@ref)
  - [`CrossSectionalWinsoriser`](@ref)
  - [`CrossSectionalTanhShrinker`](@ref)
  - [`CrossSectionalStandardiser`](@ref)
  - [`CrossSectionalGaussianRank`](@ref)
  - [`CrossSectionalPercentileRank`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
abstract type AbstractCrossSectionalTransform <: AbstractEstimator end
"""
    const CS_MISSING_GROUP = -1

The group label that says an asset carries no group at an observation.

A cell labelled this way never joins a group statistic, and it takes the whole-row statistics instead. Every other label is a group of its own, so the labels need not be contiguous and need not repeat between observations.

# Related

  - [`cross_sectional_transform`](@ref)
  - [`cross_sectional_groups`](@ref)
"""
const CS_MISSING_GROUP = -1
"""
    const CS_MAD_CONSISTENCY = 1.4826022185056018

The factor that makes a median absolute deviation consistent with a standard deviation under normality.

It is the reciprocal of the third quartile of the standard normal distribution, so a normal sample's scaled median absolute deviation estimates the same quantity its standard deviation does.

# Related

  - [`CrossSectionalTanhShrinker`](@ref)
"""
const CS_MAD_CONSISTENCY = 1.4826022185056018
"""
    assert_cross_sectional_matrix(X::MatNum) -> nothing

Check that a cross-sectional data matrix is non-empty and holds no infinite cell.

A `NaN` is the marker for a missing cell, so it is admitted and preserved. An infinity is neither a value nor a marker: it survives a quantile, a median and a mean, and it turns every statistic of its observation into an infinity or a `NaN` without saying why.

# Algorithm

 1. Refuse an empty matrix.
 2. Find the first cell that is neither finite nor `NaN`, and refuse it by name.

# Arguments

  - `X`: Data matrix `observations × assets`.

# Validation

  - `!isempty(X)`. Raises an `IsEmptyError`.
  - Every cell of `X` is finite or `NaN`. Raises a `DomainError` naming the observation and the asset.

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_transform`](@ref)
  - [`assert_cross_sectional_weights`](@ref)
  - [`assert_cross_sectional_groups`](@ref)
"""
function assert_cross_sectional_matrix(X::MatNum)::Nothing
    assert_nonempty(X, :X)
    idx = findfirst(x -> !isfinite(x) && !isnan(x), X)
    @argcheck(isnothing(idx),
              DomainError(idx,
                          "X may hold a NaN, which marks a missing cell, but not an infinity: an infinity survives every cross-sectional statistic and turns its observation into an infinity or a NaN. The first offending cell is at $(isnothing(idx) ? "" : string(Tuple(idx)))"))
    return nothing
end
"""
    assert_cross_sectional_weights(X::MatNum, w::Nothing) -> nothing
    assert_cross_sectional_weights(X::MatNum, w::MatNum) -> nothing

Check a benchmark weight matrix against the data matrix it selects the estimation set of.

A weight is a selector first and a weight second: a positive weight puts the cell in the estimation set of its observation, and a zero weight leaves it out. A `NaN` weight therefore says neither, and a negative weight says less than nothing.

# Algorithm

 1. Return when `w` is `nothing`, because the estimation set is then the finite cells alone.
 2. Check the shape against `X`, then the finiteness and the sign of every weight.

# Arguments

  - `X`: Data matrix `observations × assets`.
  - `w`: Benchmark weight matrix `observations × assets`, or `nothing`.

# Validation

  - `size(w) == size(X)`. Raises a `DimensionMismatch`.
  - Every weight is finite. Raises an `IsNonFiniteError`.
  - Every weight is non-negative. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_transform`](@ref)
  - [`assert_cross_sectional_matrix`](@ref)
  - [`assert_cross_sectional_groups`](@ref)
"""
function assert_cross_sectional_weights(::MatNum, ::Nothing)::Nothing
    return nothing
end
function assert_cross_sectional_weights(X::MatNum, w::MatNum)::Nothing
    @argcheck(size(w) == size(X),
              DimensionMismatch("the benchmark weights (w) select the estimation set of each observation, so they are observations × assets like X, got size(w) = $(size(w)) and size(X) = $(size(X))"))
    assert_all_finite(w, :w)
    assert_nonneg(w, :w)
    return nothing
end
"""
    assert_cross_sectional_groups(X::MatNum, groups::Nothing) -> nothing
    assert_cross_sectional_groups(X::MatNum, groups::AbstractMatrix{<:Integer}) -> nothing

Check a group label matrix against the data matrix it partitions.

A label is an identity, not a quantity, so only [`CS_MISSING_GROUP`](@ref) carries a meaning of its own. A label below it names no group and no missing cell, so it would silently join a partition of its own.

# Algorithm

 1. Return when `groups` is `nothing`, because every cell then takes the whole-row statistics.
 2. Check the shape against `X`, then that no label sits below [`CS_MISSING_GROUP`](@ref).

# Arguments

  - `X`: Data matrix `observations × assets`.
  - `groups`: Group label matrix `observations × assets`, or `nothing`.

# Validation

  - `size(groups) == size(X)`. Raises a `DimensionMismatch`.
  - Every label is at least [`CS_MISSING_GROUP`](@ref). Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_transform`](@ref)
  - [`cross_sectional_groups`](@ref)
  - [`CS_MISSING_GROUP`](@ref)
"""
function assert_cross_sectional_groups(::MatNum, ::Nothing)::Nothing
    return nothing
end
function assert_cross_sectional_groups(X::MatNum,
                                       groups::AbstractMatrix{<:Integer})::Nothing
    @argcheck(size(groups) == size(X),
              DimensionMismatch("the group labels (groups) partition each observation, so they are observations × assets like X, got size(groups) = $(size(groups)) and size(X) = $(size(X))"))
    idx = findfirst(g -> g < CS_MISSING_GROUP, groups)
    @argcheck(isnothing(idx),
              DomainError(idx,
                          "a group label is an identity, so the only label with a meaning of its own is CS_MISSING_GROUP = $(CS_MISSING_GROUP), and no label may sit below it. The first offending cell is at $(isnothing(idx) ? "" : string(Tuple(idx)))"))
    return nothing
end
"""
    cross_sectional_estimation_mask(fin::AbstractMatrix{Bool}, w::Nothing) -> AbstractMatrix{Bool}
    cross_sectional_estimation_mask(fin::AbstractMatrix{Bool}, w::MatNum) -> AbstractMatrix{Bool}

Return the estimation mask of every observation of a cross-sectional transform.

The estimation set of an observation is what its statistics are computed from. Without benchmark weights it is the finite cells, and with them it is the finite cells carrying a positive weight. A cell outside the set is still transformed against it, so the mask selects the estimator and never the output.

# Algorithm

 1. Return the finiteness mask when `w` is `nothing`.
 2. Otherwise intersect it with the cells of positive weight.

# Arguments

  - `fin::AbstractMatrix{Bool}`: Finiteness mask `observations × assets`.
  - `w`: Benchmark weight matrix `observations × assets`, or `nothing`.

# Returns

  - `est::AbstractMatrix{Bool}`: Estimation mask `observations × assets`. It is `fin` itself when `w` is `nothing`, and neither mask is written to.

# Related

  - [`cross_sectional_transform`](@ref)
  - [`cross_sectional_indices`](@ref)
"""
function cross_sectional_estimation_mask(fin::AbstractMatrix{Bool}, ::Nothing)
    return fin
end
function cross_sectional_estimation_mask(fin::AbstractMatrix{Bool}, w::MatNum)
    return fin .& (w .> zero(eltype(w)))
end
"""
    cross_sectional_indices(msk::AbstractMatrix{Bool}, t::Integer) -> Vector{Int}

Return the asset indices an `observations × assets` mask selects at one observation.

# Arguments

  - `msk::AbstractMatrix{Bool}`: Mask `observations × assets`.
  - `t::Integer`: Observation.

# Returns

  - `idx::Vector{Int}`: Selected asset indices, in ascending order.

# Examples

```jldoctest
julia> PortfolioOptimisers.cross_sectional_indices([true false true; false true false], 1)
2-element Vector{Int64}:
 1
 3
```

# Related

  - [`cross_sectional_estimation_mask`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
function cross_sectional_indices(msk::AbstractMatrix{Bool}, t::Integer)::Vector{Int}
    return [i for i in axes(msk, 2) if msk[t, i]]
end
"""
    cross_sectional_weight_type(w::Nothing) -> Type
    cross_sectional_weight_type(w::MatNum) -> Type

Return the element type a benchmark weight matrix contributes to the output of a transform.

An absent weight matrix contributes `Bool`, which promotes against every numeric type without widening it, so the caller needs no branch of its own.

# Arguments

  - `w`: Benchmark weight matrix `observations × assets`, or `nothing`.

# Returns

  - `T::Type`: `Bool` when `w` is `nothing`, and `eltype(w)` otherwise.

# Related

  - [`cross_sectional_transform`](@ref)
"""
function cross_sectional_weight_type(::Nothing)
    return Bool
end
function cross_sectional_weight_type(w::MatNum)
    return eltype(w)
end
"""
    cross_sectional_weighted_mean(A::AbstractMatrix, w::Nothing, t::Integer, idx::AbstractVector{<:Integer})
    cross_sectional_weighted_mean(A::AbstractMatrix, w::MatNum, t::Integer, idx::AbstractVector{<:Integer})

Return the centre of one observation's estimation set.

The centre is weighted when benchmark weights are given, so a large benchmark holding pulls it towards itself, and it is the plain mean otherwise.

# Arguments

  - `A::AbstractMatrix`: Data matrix `observations × assets`, already floating point.
  - `w`: Benchmark weight matrix `observations × assets`, or `nothing`.
  - `t::Integer`: Observation.
  - `idx::AbstractVector{<:Integer}`: Estimation set of the observation, which must not be empty.

# Returns

  - `mu::Number`: The centre, in the element type of `A`. It is zero when every weight of the estimation set is zero, which the estimation set's own definition excludes.

# Related

  - [`cross_sectional_equal_std`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
function cross_sectional_weighted_mean(A::AbstractMatrix, ::Nothing, t::Integer,
                                       idx::AbstractVector{<:Integer})
    s = zero(eltype(A))
    for i in idx
        s += A[t, i]
    end
    return s / length(idx)
end
function cross_sectional_weighted_mean(A::AbstractMatrix, w::MatNum, t::Integer,
                                       idx::AbstractVector{<:Integer})
    sw = zero(eltype(A))
    sx = zero(eltype(A))
    for i in idx
        sw += w[t, i]
        sx += w[t, i] * A[t, i]
    end
    return iszero(sw) ? zero(eltype(A)) : sx / sw
end
"""
    cross_sectional_equal_std(A::AbstractMatrix, t::Integer, idx::AbstractVector{<:Integer}, mu::Number)

Return the equal-weighted dispersion of one observation's estimation set around a centre.

The scale is equal-weighted even where the centre is weighted, because a benchmark weight says how much of the market an asset is and not how precisely its value is measured. The divisor is the sample size less one.

# Arguments

  - `A::AbstractMatrix`: Data matrix `observations × assets`, already floating point.
  - `t::Integer`: Observation.
  - `idx::AbstractVector{<:Integer}`: Estimation set of the observation.
  - `mu::Number`: Centre to measure the dispersion around.

# Returns

  - `sigma::Number`: The dispersion, in the element type of `A`. It is zero when the estimation set holds fewer than two assets, and every consumer then reads the cell as having no dispersion rather than an undefined one.

# Related

  - [`cross_sectional_weighted_mean`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
function cross_sectional_equal_std(A::AbstractMatrix, t::Integer,
                                   idx::AbstractVector{<:Integer}, mu::Number)
    if length(idx) < 2
        return zero(eltype(A))
    end
    ss = zero(eltype(A))
    for i in idx
        d = A[t, i] - mu
        ss += d * d
    end
    return sqrt(ss / (length(idx) - 1))
end
"""
    cross_sectional_stat(v::AbstractVector, i::Integer) -> Number
    cross_sectional_stat(v::Number, i::Integer) -> Number

Read the statistic that applies to one asset, whether it is per asset or shared by the observation.

An ungrouped transform holds one centre and one scale for the whole observation, and a grouped one holds a pair per asset. This reader lets one scoring loop serve both.

# Arguments

  - `v`: Statistic, either one number per asset or one number for the observation.
  - `i::Integer`: Asset.

# Returns

  - `val::Number`: The statistic of asset `i`.

# Related

  - [`cross_sectional_zscore_row!`](@ref)
"""
function cross_sectional_stat(v::AbstractVector, i::Integer)
    return v[i]
end
function cross_sectional_stat(v::Number, ::Integer)
    return v
end
"""
    cross_sectional_blank_row!(Y::AbstractMatrix, t::Integer) -> nothing

Write a missing marker into every asset of one observation.

An observation whose estimation set is empty has nothing to be transformed against, so no cell of it carries a value, not even a cell that was finite.

# Arguments

  - `Y::AbstractMatrix`: Output matrix `observations × assets`.
  - `t::Integer`: Observation.

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_transform`](@ref)
"""
function cross_sectional_blank_row!(Y::AbstractMatrix, t::Integer)::Nothing
    for i in axes(Y, 2)
        Y[t, i] = NaN
    end
    return nothing
end
"""
    cross_sectional_zscore_row!(Y::AbstractMatrix, A::AbstractMatrix, fin::AbstractMatrix{Bool}, t::Integer, idx::AbstractVector{<:Integer}, mu, sigma, atol::Real) -> nothing

Score one observation against a centre and a scale.

An observation with no dispersion scores every cell zero rather than dividing by it, so a cross-section that carries the same value everywhere reads as a neutral exposure instead of a missing one.

# Algorithm

 1. Blank the whole observation when its estimation set is empty.
 2. Otherwise write a missing marker at every non-finite cell.
 3. Write the centred and rescaled value where the scale is above `atol`, and zero where it is not.

# Arguments

  - `Y::AbstractMatrix`: Output matrix `observations × assets`.
  - `A::AbstractMatrix`: Data matrix `observations × assets`, already floating point.
  - `fin::AbstractMatrix{Bool}`: Finiteness mask `observations × assets`.
  - `t::Integer`: Observation.
  - `idx::AbstractVector{<:Integer}`: Estimation set of the observation.
  - `mu`: Centre, one number for the observation or one per asset.
  - `sigma`: Scale, one number for the observation or one per asset.
  - `atol::Real`: $(arg_dict[:atol_cs])

# Returns

  - `nothing`.

# Related

  - [`cross_sectional_stat`](@ref)
  - [`CrossSectionalStandardiser`](@ref)
"""
function cross_sectional_zscore_row!(Y::AbstractMatrix, A::AbstractMatrix,
                                     fin::AbstractMatrix{Bool}, t::Integer,
                                     idx::AbstractVector{<:Integer}, mu, sigma,
                                     atol::Real)::Nothing
    if isempty(idx)
        return cross_sectional_blank_row!(Y, t)
    end
    for i in axes(Y, 2)
        s = cross_sectional_stat(sigma, i)
        if !fin[t, i]
            Y[t, i] = NaN
        elseif s > atol
            Y[t, i] = (A[t, i] - cross_sectional_stat(mu, i)) / s
        else
            Y[t, i] = zero(eltype(Y))
        end
    end
    return nothing
end
"""
    cross_sectional_recentre_rescale!(Y::AbstractMatrix, fin::AbstractMatrix{Bool}, est::AbstractMatrix{Bool}, w::Option{<:MatNum}, atol::Real, scale::Bool) -> nothing

Recentre, and optionally rescale, an already scored matrix over the whole cross-section.

A grouped score is comparable inside its group and not between groups, so the score is brought back to a weighted centre of zero over the whole observation, and to a unit equal-weighted scale when the caller asks for one.

# Algorithm

 1. Blank an observation whose estimation set is empty.
 2. Subtract the weighted centre of the estimation set from every finite cell.
 3. Return when `scale` is `false`.
 4. Divide by the equal-weighted scale of the estimation set, writing zero where that scale is at or below `atol`.

# Arguments

  - `Y::AbstractMatrix`: Score matrix `observations × assets`, written in place.
  - `fin::AbstractMatrix{Bool}`: Finiteness mask `observations × assets` of the matrix the score came from.
  - `est::AbstractMatrix{Bool}`: Estimation mask `observations × assets`.
  - `w`: Benchmark weight matrix `observations × assets`, or `nothing`.
  - `atol::Real`: $(arg_dict[:atol_cs])
  - `scale::Bool`: Whether to divide by the equal-weighted scale.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalStandardiser`](@ref)
  - [`CrossSectionalGaussianRank`](@ref)
"""
function cross_sectional_recentre_rescale!(Y::AbstractMatrix, fin::AbstractMatrix{Bool},
                                           est::AbstractMatrix{Bool}, w::Option{<:MatNum},
                                           atol::Real, scale::Bool)::Nothing
    for t in axes(Y, 1)
        idx = cross_sectional_indices(est, t)
        if isempty(idx)
            cross_sectional_blank_row!(Y, t)
            continue
        end
        mu = cross_sectional_weighted_mean(Y, w, t, idx)
        for i in axes(Y, 2)
            Y[t, i] = fin[t, i] ? Y[t, i] - mu : Y[t, i]
        end
        if scale
            s = cross_sectional_equal_std(Y, t, idx, zero(eltype(Y)))
            cross_sectional_zscore_row!(Y, Y, fin, t, idx, zero(eltype(Y)), s, atol)
        end
    end
    return nothing
end
"""
    cross_sectional_midranks!(P::AbstractMatrix, A::AbstractMatrix, t::Integer, idx::AbstractVector{<:Integer}, qry::AbstractVector{<:Integer}) -> nothing

Write the percentile rank of each queried asset against one estimation set.

A tie shares the average of the ranks its members would otherwise occupy, and the rank is centred inside its bin, so a percentile sits strictly inside the open unit interval and an inverse normal of it is always finite.

# Algorithm

 1. Write a missing marker at every queried asset when the estimation set is empty.
 2. Otherwise sort the estimation values once.
 3. For each queried asset, count the estimation values strictly below it and those at or below it, and average the two counts.
 4. Divide by the size of the estimation set, and clamp into the closed interval between half a bin and one less half a bin.

# Arguments

  - `P::AbstractMatrix`: Percentile matrix `observations × assets`, written in place.
  - `A::AbstractMatrix`: Data matrix `observations × assets`, already floating point.
  - `t::Integer`: Observation.
  - `idx::AbstractVector{<:Integer}`: Estimation set to rank against.
  - `qry::AbstractVector{<:Integer}`: Assets to rank.

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalPercentileRank`](@ref)
  - [`cross_sectional_percentile_ranks`](@ref)
"""
function cross_sectional_midranks!(P::AbstractMatrix, A::AbstractMatrix, t::Integer,
                                   idx::AbstractVector{<:Integer},
                                   qry::AbstractVector{<:Integer})::Nothing
    n = length(idx)
    if iszero(n)
        for i in qry
            P[t, i] = NaN
        end
        return nothing
    end
    v = sort!([A[t, i] for i in idx])
    lo = one(eltype(P)) / (2 * n)
    for i in qry
        nlt, nle = cross_sectional_rank_counts(v, A[t, i])
        P[t, i] = clamp((nlt + nle) / (2 * n), lo, one(lo) - lo)
    end
    return nothing
end
"""
    cross_sectional_rank_counts(v::AbstractVector, x::Number) -> Tuple{Int, Int}

Return how many entries of a sorted vector sit below a value, and how many sit at or below it.

The pair is what a midrank needs: their average is the rank a tie shares. Two binary searches read
it in logarithmic time, and the second starts where the first stopped, because a value at or below
`x` is never below one that is below it.

The searches are written out rather than taken from `Base`, because a loaded dependency adds its own
methods to `searchsortedfirst` and `searchsortedlast`, and the static analysis gate reads those arms
and refuses the call.

# Algorithm

 1. Binary search for the largest prefix of `v` whose entries are all below `x`.
 2. Binary search again, from that prefix to the end, for the largest prefix whose entries are all at
    or below `x`.

# Arguments

  - `v::AbstractVector`: Values to count against, sorted in ascending order.
  - `x::Number`: Value to count around.

# Returns

  - `nlt::Int`: Number of entries strictly below `x`.
  - `nle::Int`: Number of entries at or below `x`.

# Examples

```jldoctest
julia> PortfolioOptimisers.cross_sectional_rank_counts([1.0, 2.0, 2.0, 3.0], 2.0)
(1, 3)

julia> PortfolioOptimisers.cross_sectional_rank_counts([1.0, 2.0, 2.0, 3.0], 0.5)
(0, 0)
```

# Related

  - [`cross_sectional_midranks!`](@ref)
  - [`CrossSectionalPercentileRank`](@ref)
"""
function cross_sectional_rank_counts(v::AbstractVector, x::Number)::Tuple{Int, Int}
    lo = 0
    hi = length(v)
    while lo < hi
        mid = (lo + hi + 1) >> 1
        if v[mid] < x
            lo = mid
        else
            hi = mid - 1
        end
    end
    nlt = lo
    hi = length(v)
    while lo < hi
        mid = (lo + hi + 1) >> 1
        if v[mid] <= x
            lo = mid
        else
            hi = mid - 1
        end
    end
    return nlt, lo
end
"""
    cross_sectional_row_groups(est::AbstractMatrix{Bool}, groups::AbstractMatrix{<:Integer}, t::Integer) -> Dict{Int, Vector{Int}}

Return the estimation set of each group of one observation.

A cell labelled [`CS_MISSING_GROUP`](@ref) joins no group, so it contributes to no group statistic while still belonging to the observation's own estimation set.

# Arguments

  - `est::AbstractMatrix{Bool}`: Estimation mask `observations × assets`.
  - `groups::AbstractMatrix{<:Integer}`: Group label matrix `observations × assets`.
  - `t::Integer`: Observation.

# Returns

  - `gidx::Dict{Int, Vector{Int}}`: Estimation asset indices of each label present at the observation, in ascending order.

# Related

  - [`cross_sectional_transform`](@ref)
  - [`CS_MISSING_GROUP`](@ref)
"""
function cross_sectional_row_groups(est::AbstractMatrix{Bool},
                                    groups::AbstractMatrix{<:Integer},
                                    t::Integer)::Dict{Int, Vector{Int}}
    gidx = Dict{Int, Vector{Int}}()
    for i in axes(est, 2)
        g = Int(groups[t, i])
        if est[t, i] && g != CS_MISSING_GROUP
            push!(get!(() -> Int[], gidx, g), i)
        end
    end
    return gidx
end
"""
    cross_sectional_group_split(fin::AbstractMatrix{Bool}, groups::AbstractMatrix{<:Integer}, t::Integer, gidx::Dict{Int, Vector{Int}}, mgs::Integer)

Split the finite assets of one observation into the groups that stand and the ones that fall back.

A group stands when its own estimation set is large enough to estimate from. Every other finite asset takes the whole-row statistics, which is what the fallback list carries.

# Algorithm

 1. Walk the finite assets of the observation.
 2. Send an asset whose label is [`CS_MISSING_GROUP`](@ref), or whose group holds fewer than `mgs` estimation assets, to the fallback list.
 3. Send every other asset to the query list of its own group.

# Arguments

  - `fin::AbstractMatrix{Bool}`: Finiteness mask `observations × assets`.
  - `groups::AbstractMatrix{<:Integer}`: Group label matrix `observations × assets`.
  - `t::Integer`: Observation.
  - `gidx::Dict{Int, Vector{Int}}`: Estimation set of each group of the observation.
  - `mgs::Integer`: $(arg_dict[:min_group_size])

# Returns

  - `qry::Dict{Int, Vector{Int}}`: Finite assets of each group that stands.
  - `fb::Vector{Int}`: Finite assets that take the whole-row statistics.

# Related

  - [`cross_sectional_row_groups`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
function cross_sectional_group_split(fin::AbstractMatrix{Bool},
                                     groups::AbstractMatrix{<:Integer}, t::Integer,
                                     gidx::Dict{Int, Vector{Int}}, mgs::Integer)
    qry = Dict{Int, Vector{Int}}()
    fb = Int[]
    for i in axes(fin, 2)
        if !fin[t, i]
            continue
        end
        g = Int(groups[t, i])
        n = haskey(gidx, g) ? length(gidx[g]) : 0
        if n < mgs
            push!(fb, i)
        else
            push!(get!(() -> Int[], qry, g), i)
        end
    end
    return qry, fb
end
"""
    cross_sectional_percentile_ranks(A::AbstractMatrix, fin::AbstractMatrix{Bool}, est::AbstractMatrix{Bool}, groups::Nothing, mgs::Integer)
    cross_sectional_percentile_ranks(A::AbstractMatrix, fin::AbstractMatrix{Bool}, est::AbstractMatrix{Bool}, groups::AbstractMatrix{<:Integer}, mgs::Integer)

Return the percentile rank of every finite cell of a matrix, against its own cross-section.

Without group labels a cell is ranked against the whole estimation set of its observation. With them it is ranked inside its own group, and it falls back to the whole estimation set when its group is too small to rank inside.

# Algorithm

 1. Walk the observations.
 2. Rank every finite asset against the estimation set of the observation when no labels are given.
 3. Otherwise split the finite assets into the groups that stand and the ones that fall back, rank each standing group against its own estimation set, and rank the fallback list against the whole estimation set.

# Arguments

  - `A::AbstractMatrix`: Data matrix `observations × assets`, already floating point.
  - `fin::AbstractMatrix{Bool}`: Finiteness mask `observations × assets`.
  - `est::AbstractMatrix{Bool}`: Estimation mask `observations × assets`.
  - `groups`: Group label matrix `observations × assets`, or `nothing`.
  - `mgs::Integer`: $(arg_dict[:min_group_size])

# Returns

  - `P::Matrix{<:AbstractFloat}`: Percentile matrix `observations × assets`, carrying a missing marker at every cell that is not finite and at every cell whose estimation set is empty.

# Related

  - [`CrossSectionalPercentileRank`](@ref)
  - [`CrossSectionalGaussianRank`](@ref)
  - [`cross_sectional_midranks!`](@ref)
"""
function cross_sectional_percentile_ranks(A::AbstractMatrix, fin::AbstractMatrix{Bool},
                                          est::AbstractMatrix{Bool}, ::Nothing, ::Integer)
    P = fill(convert(eltype(A), NaN), size(A))
    for t in axes(A, 1)
        cross_sectional_midranks!(P, A, t, cross_sectional_indices(est, t),
                                  cross_sectional_indices(fin, t))
    end
    return P
end
function cross_sectional_percentile_ranks(A::AbstractMatrix, fin::AbstractMatrix{Bool},
                                          est::AbstractMatrix{Bool},
                                          groups::AbstractMatrix{<:Integer}, mgs::Integer)
    P = fill(convert(eltype(A), NaN), size(A))
    for t in axes(A, 1)
        idx = cross_sectional_indices(est, t)
        gidx = cross_sectional_row_groups(est, groups, t)
        qry, fb = cross_sectional_group_split(fin, groups, t, gidx, mgs)
        for (g, cols) in qry
            cross_sectional_midranks!(P, A, t, gidx[g], cols)
        end
        cross_sectional_midranks!(P, A, t, idx, fb)
    end
    return P
end
"""
$(DocStringExtensions.TYPEDEF)

Clips every value of an observation into the band between two percentiles of that observation's cross-section.

The clip is a hard one, so a value beyond a band edge takes the edge itself and every value inside the band is untouched. It is the cheapest way to stop one asset's extreme reading from dominating a cross-sectional fit.

# Mathematical definition

```math
\\begin{align}
x_{t,i}' &= \\min\\left(\\max\\left(x_{t,i},\\, q_{t}^{\\mathrm{lo}}\\right),\\, q_{t}^{\\mathrm{hi}}\\right)\\,.
\\end{align}
```

Where:

  - ``x_{t,i}``: Value of asset ``i`` at observation ``t``.
  - ``q_{t}^{\\mathrm{lo}}``, ``q_{t}^{\\mathrm{hi}}``: Percentiles of the estimation set of observation ``t``, at the levels `low` and `high`.

The percentiles are equal-weighted over the estimation set. An asset outside that set is still clipped to the same band, and a `NaN` stays a `NaN`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalWinsoriser(; low::Real = 0.01, high::Real = 0.99) -> CrossSectionalWinsoriser

Keywords correspond to the struct's fields.

## Validation

  - `0 <= low < high <= 1`.

# Examples

```jldoctest
julia> CrossSectionalWinsoriser()
CrossSectionalWinsoriser
   low ┼ Float64: 0.01
  high ┴ Float64: 0.99
```

# Related

  - [`AbstractCrossSectionalTransform`](@ref)
  - [`CrossSectionalTanhShrinker`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
@concrete struct CrossSectionalWinsoriser <: AbstractCrossSectionalTransform
    """
    Percentile level of the lower edge of the band.
    """
    low
    """
    Percentile level of the upper edge of the band.
    """
    high
    function CrossSectionalWinsoriser(low::Real, high::Real)
        @argcheck(zero(low) <= low < high <= one(high),
                  DomainError((low, high),
                              "low and high must satisfy 0 <= low < high <= 1, got low = $(low) and high = $(high)"))
        return new{typeof(low), typeof(high)}(low, high)
    end
end
function CrossSectionalWinsoriser(; low::Real = 0.01,
                                  high::Real = 0.99)::CrossSectionalWinsoriser
    return CrossSectionalWinsoriser(low, high)
end
"""
$(DocStringExtensions.TYPEDEF)

Compresses every value of an observation towards the centre of that observation's cross-section, through a hyperbolic tangent.

The map is smooth and strictly increasing, so it creates no jump at a threshold and it keeps the order of the tail. A value near the centre moves almost not at all, and a value far from it is pulled in hard. The output keeps the units of the input.

# Mathematical definition

```math
\\begin{align}
x_{t,i}' &= m_{t} + h_{t} \\tanh\\left(\\frac{x_{t,i} - m_{t}}{h_{t}}\\right) \\\\
h_{t} &= c \\, \\gamma \\, \\mathrm{med}_{j}\\left(\\left\\lvert x_{t,j} - m_{t} \\right\\rvert\\right)\\,.
\\end{align}
```

Where:

  - ``x_{t,i}``: Value of asset ``i`` at observation ``t``.
  - ``m_{t}``: Median of the estimation set of observation ``t``.
  - ``\\gamma``: The constant `1.4826…`, which makes a median absolute deviation consistent with a standard deviation under normality.
  - ``c``: Compression knee.
  - ``h_{t}``: Half-width of the near-linear region of observation ``t``.

The median and the median absolute deviation are equal-weighted over the estimation set. An asset outside that set is still compressed against the same statistics, and a `NaN` stays a `NaN`. An observation whose robust scale is at or below `atol` is returned unchanged, because it carries no dispersion to compress against.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalTanhShrinker(; knee::Real = 3.0, atol::Real = 1e-12) -> CrossSectionalTanhShrinker

Keywords correspond to the struct's fields.

## Validation

  - `isfinite(knee)` and `knee > 0`.
  - `isfinite(atol)` and `atol >= 0`.

# Examples

```jldoctest
julia> CrossSectionalTanhShrinker()
CrossSectionalTanhShrinker
  knee ┼ Float64: 3.0
  atol ┴ Float64: 1.0e-12
```

# Related

  - [`AbstractCrossSectionalTransform`](@ref)
  - [`CrossSectionalWinsoriser`](@ref)
  - [`cross_sectional_transform`](@ref)
  - [`CS_MAD_CONSISTENCY`](@ref)
"""
@concrete struct CrossSectionalTanhShrinker <: AbstractCrossSectionalTransform
    """
    Half-width of the near-linear region, counted in robust standard deviations. A larger knee compresses less.
    """
    knee
    """
    $(field_dict[:atol_cs])
    """
    atol
    function CrossSectionalTanhShrinker(knee::Real, atol::Real)
        assert_finite(knee, :knee)
        assert_gt0(knee, :knee)
        assert_finite(atol, :atol)
        assert_nonneg(atol, :atol)
        return new{typeof(knee), typeof(atol)}(knee, atol)
    end
end
function CrossSectionalTanhShrinker(; knee::Real = 3.0,
                                    atol::Real = 1e-12)::CrossSectionalTanhShrinker
    return CrossSectionalTanhShrinker(knee, atol)
end
"""
$(DocStringExtensions.TYPEDEF)

Scores every value of an observation as a cross-sectional z-score, optionally inside its own group first.

The centre is weighted and the scale is equal-weighted, so the score says how far an asset sits from the benchmark's own centre, measured in the dispersion of the cross-section. A grouped score is then brought back to a common centre and scale, so scores from different groups are comparable.

# Mathematical definition

```math
\\begin{align}
z_{t,i} &= \\frac{x_{t,i} - \\mu_{t}}{\\sigma_{t}} \\\\
\\mu_{t} &= \\frac{\\sum_{j \\in \\mathcal{E}_{t}} w_{t,j} x_{t,j}}{\\sum_{j \\in \\mathcal{E}_{t}} w_{t,j}} \\\\
\\sigma_{t} &= \\sqrt{\\frac{1}{\\lvert \\mathcal{E}_{t} \\rvert - 1} \\sum_{j \\in \\mathcal{E}_{t}} \\left(x_{t,j} - \\mu_{t}\\right)^{2}}\\,.
\\end{align}
```

Where:

  - ``x_{t,i}``: Value of asset ``i`` at observation ``t``.
  - ``w_{t,j}``: Benchmark weight of asset ``j`` at observation ``t``.
  - ``\\mathcal{E}_{t}``: Estimation set of observation ``t``.
  - ``\\mu_{t}``, ``\\sigma_{t}``: Centre and scale of observation ``t``.

With group labels the same pair is computed inside each group, and the score is recentred and rescaled over the whole observation afterwards. A group holding fewer than `min_group_size` estimation assets, and every asset labelled [`CS_MISSING_GROUP`](@ref), takes the whole-observation pair instead. An observation whose scale is at or below `atol` scores zero, and a `NaN` stays a `NaN`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalStandardiser(; min_group_size::Integer = 8, atol::Real = 1e-12) -> CrossSectionalStandardiser

Keywords correspond to the struct's fields.

## Validation

  - `min_group_size >= 1`.
  - `isfinite(atol)` and `atol >= 0`.

# Examples

```jldoctest
julia> CrossSectionalStandardiser()
CrossSectionalStandardiser
  min_group_size ┼ Int64: 8
            atol ┴ Float64: 1.0e-12
```

# Related

  - [`AbstractCrossSectionalTransform`](@ref)
  - [`CrossSectionalGaussianRank`](@ref)
  - [`CrossSectionalPercentileRank`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
@concrete struct CrossSectionalStandardiser <: AbstractCrossSectionalTransform
    """
    $(field_dict[:min_group_size])
    """
    min_group_size
    """
    $(field_dict[:atol_cs])
    """
    atol
    function CrossSectionalStandardiser(min_group_size::Integer, atol::Real)
        assert_gt0(min_group_size, :min_group_size)
        assert_finite(atol, :atol)
        assert_nonneg(atol, :atol)
        return new{typeof(min_group_size), typeof(atol)}(min_group_size, atol)
    end
end
function CrossSectionalStandardiser(; min_group_size::Integer = 8,
                                    atol::Real = 1e-12)::CrossSectionalStandardiser
    return CrossSectionalStandardiser(min_group_size, atol)
end
"""
$(DocStringExtensions.TYPEDEF)

Scores every value of an observation by the inverse normal of its cross-sectional percentile rank.

The score depends on the order of the cross-section and not on the size of its gaps, so one extreme reading moves no other asset's score. The inverse normal spreads the ranks the way a normal sample would be spread, which is what makes the output usable as a factor exposure.

# Mathematical definition

```math
\\begin{align}
z_{t,i} &= \\frac{\\Phi^{-1}\\left(p_{t,i}\\right) - \\mu_{t}}{\\sigma_{t}}\\,.
\\end{align}
```

Where:

  - ``p_{t,i}``: Percentile rank of asset ``i`` at observation ``t``, as [`CrossSectionalPercentileRank`](@ref) defines it.
  - ``\\Phi^{-1}``: Inverse cumulative distribution function of the standard normal distribution.
  - ``\\mu_{t}``: Weighted centre of the inverse normals of observation ``t``.
  - ``\\sigma_{t}``: Equal-weighted scale of the inverse normals of observation ``t``, which is divided by only when `scale` is `true`.

The ranking is equal-weighted over the estimation set, and the recentring is weighted. The recentring and the rescaling always run over the whole observation, never inside a group. An observation whose scale is at or below `atol` scores zero after the recentring, and a `NaN` stays a `NaN`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalGaussianRank(; min_group_size::Integer = 8, scale::Bool = true, atol::Real = 1e-12) -> CrossSectionalGaussianRank

Keywords correspond to the struct's fields.

## Validation

  - `min_group_size >= 1`.
  - `isfinite(atol)` and `atol >= 0`.

# Examples

```jldoctest
julia> CrossSectionalGaussianRank()
CrossSectionalGaussianRank
  min_group_size ┼ Int64: 8
           scale ┼ Bool: true
            atol ┴ Float64: 1.0e-12
```

# Related

  - [`AbstractCrossSectionalTransform`](@ref)
  - [`CrossSectionalPercentileRank`](@ref)
  - [`CrossSectionalStandardiser`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
@concrete struct CrossSectionalGaussianRank <: AbstractCrossSectionalTransform
    """
    $(field_dict[:min_group_size])
    """
    min_group_size
    """
    Whether to divide the recentred scores by their equal-weighted scale. Leave it `false` to feed a scale-invariant consumer, which then reads no dispersion noise from the estimate of that scale.
    """
    scale
    """
    $(field_dict[:atol_cs])
    """
    atol
    function CrossSectionalGaussianRank(min_group_size::Integer, scale::Bool, atol::Real)
        assert_gt0(min_group_size, :min_group_size)
        assert_finite(atol, :atol)
        assert_nonneg(atol, :atol)
        return new{typeof(min_group_size), typeof(scale), typeof(atol)}(min_group_size,
                                                                        scale, atol)
    end
end
function CrossSectionalGaussianRank(; min_group_size::Integer = 8, scale::Bool = true,
                                    atol::Real = 1e-12)::CrossSectionalGaussianRank
    return CrossSectionalGaussianRank(min_group_size, scale, atol)
end
"""
$(DocStringExtensions.TYPEDEF)

Scores every value of an observation by its percentile rank inside that observation's cross-section.

The rank is centred inside its own bin, so the score sits strictly inside the open unit interval and an inverse normal of it is always finite. A tie shares the average of the ranks its members would otherwise occupy.

# Mathematical definition

```math
\\begin{align}
p_{t,i} &= \\mathrm{clamp}\\left(\\frac{\\#\\left\\{j \\in \\mathcal{E}_{t} : x_{t,j} < x_{t,i}\\right\\} + \\#\\left\\{j \\in \\mathcal{E}_{t} : x_{t,j} \\le x_{t,i}\\right\\}}{2 \\lvert \\mathcal{E}_{t} \\rvert},\\, \\frac{1}{2 \\lvert \\mathcal{E}_{t} \\rvert},\\, 1 - \\frac{1}{2 \\lvert \\mathcal{E}_{t} \\rvert}\\right)\\,.
\\end{align}
```

Where:

  - ``x_{t,i}``: Value of asset ``i`` at observation ``t``.
  - ``\\mathcal{E}_{t}``: Estimation set of observation ``t``.
  - ``p_{t,i}``: Percentile rank of asset ``i`` at observation ``t``.

The ranking is equal-weighted over the estimation set, so a benchmark weight selects that set and nothing else. With group labels an asset is ranked inside its own group. A group holding fewer than `min_group_size` estimation assets, and every asset labelled [`CS_MISSING_GROUP`](@ref), is ranked against the whole observation instead. A `NaN` stays a `NaN`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CrossSectionalPercentileRank(; min_group_size::Integer = 8) -> CrossSectionalPercentileRank

Keywords correspond to the struct's fields.

## Validation

  - `min_group_size >= 1`.

# Examples

```jldoctest
julia> CrossSectionalPercentileRank()
CrossSectionalPercentileRank
  min_group_size ┴ Int64: 8
```

# Related

  - [`AbstractCrossSectionalTransform`](@ref)
  - [`CrossSectionalGaussianRank`](@ref)
  - [`CrossSectionalStandardiser`](@ref)
  - [`cross_sectional_transform`](@ref)
"""
@concrete struct CrossSectionalPercentileRank <: AbstractCrossSectionalTransform
    """
    $(field_dict[:min_group_size])
    """
    min_group_size
    function CrossSectionalPercentileRank(min_group_size::Integer)
        assert_gt0(min_group_size, :min_group_size)
        return new{typeof(min_group_size)}(min_group_size)
    end
end
function CrossSectionalPercentileRank(;
                                      min_group_size::Integer = 8)::CrossSectionalPercentileRank
    return CrossSectionalPercentileRank(min_group_size)
end
"""
    cross_sectional_transform(ct::CrossSectionalWinsoriser, X::MatNum; w::Option{<:MatNum} = nothing, groups::Option{<:AbstractMatrix{<:Integer}} = nothing)
    cross_sectional_transform(ct::CrossSectionalTanhShrinker, X::MatNum; w::Option{<:MatNum} = nothing, groups::Option{<:AbstractMatrix{<:Integer}} = nothing)
    cross_sectional_transform(ct::CrossSectionalStandardiser, X::MatNum; w::Option{<:MatNum} = nothing, groups::Option{<:AbstractMatrix{<:Integer}} = nothing)
    cross_sectional_transform(ct::CrossSectionalGaussianRank, X::MatNum; w::Option{<:MatNum} = nothing, groups::Option{<:AbstractMatrix{<:Integer}} = nothing)
    cross_sectional_transform(ct::CrossSectionalPercentileRank, X::MatNum; w::Option{<:MatNum} = nothing, groups::Option{<:AbstractMatrix{<:Integer}} = nothing)

Transform each observation of a matrix against the other assets of that same observation.

The benchmark weights and the group labels are arguments and never fields, because one transform runs against a different benchmark and a different classification at every call site, while the transform itself is one configuration.

Four rules are shared by every member:

  - The **estimation set** of an observation is its finite cells carrying a positive weight when `w` is given, and its finite cells otherwise. It is what the statistics are computed from.
  - A cell outside the estimation set is still transformed **against** it, so an asset the benchmark does not hold is scored on the same scale as one it does.
  - An observation whose estimation set is empty returns a missing marker at every asset, because it has nothing to be transformed against.
  - The two outlier members accept `groups` and ignore it: an outlier is extreme against the whole cross-section, not against a sector.

# Algorithm

 1. Check `X`, `w` and `groups`, then promote them to one floating point element type.
 2. Build the finiteness mask and the estimation mask.
 3. Walk the observations, and apply the member's own map to each.

# Arguments

  - `ct`: Cross-sectional transform.
  - `X::MatNum`: Data matrix `observations × assets`, where a `NaN` marks a missing cell.
  - `w`: Benchmark weight matrix `observations × assets`, or `nothing`.
  - `groups`: Group label matrix `observations × assets`, or `nothing`. [`CS_MISSING_GROUP`](@ref) labels an asset that carries no group.

# Validation

  - `!isempty(X)`, and every cell of `X` is finite or `NaN`.
  - `size(w) == size(X)`, and every weight is finite and non-negative.
  - `size(groups) == size(X)`, and every label is at least [`CS_MISSING_GROUP`](@ref).

# Returns

  - `Y::Matrix{<:AbstractFloat}`: Transformed matrix `observations × assets`, carrying a missing marker wherever `X` did.

# Examples

```jldoctest
julia> X = [1.0 NaN 3.0 4.0; 4.0 3.0 2.0 1.0];

julia> cross_sectional_transform(CrossSectionalWinsoriser(; low = 0.1, high = 0.9), X)
2×4 Matrix{Float64}:
 1.4  NaN    3.0  3.8
 3.7    3.0  2.0  1.3

julia> cross_sectional_transform(CrossSectionalPercentileRank(), X)
2×4 Matrix{Float64}:
 0.166667  NaN      0.5    0.833333
 0.875       0.625  0.375  0.125
```

# Related

  - [`AbstractCrossSectionalTransform`](@ref)
  - [`CrossSectionalWinsoriser`](@ref)
  - [`CrossSectionalTanhShrinker`](@ref)
  - [`CrossSectionalStandardiser`](@ref)
  - [`CrossSectionalGaussianRank`](@ref)
  - [`CrossSectionalPercentileRank`](@ref)
  - [`cross_sectional_groups`](@ref)
"""
function cross_sectional_transform(ct::CrossSectionalWinsoriser, X::MatNum;
                                   w::Option{<:MatNum} = nothing,
                                   groups::Option{<:AbstractMatrix{<:Integer}} = nothing)
    assert_cross_sectional_matrix(X)
    assert_cross_sectional_weights(X, w)
    assert_cross_sectional_groups(X, groups)
    T = float(promote_type(eltype(X), cross_sectional_weight_type(w), typeof(ct.low),
                           typeof(ct.high)))
    A = convert(Matrix{T}, X)
    fin = isfinite.(A)
    est = cross_sectional_estimation_mask(fin, w)
    Y = similar(A)
    for t in axes(A, 1)
        idx = cross_sectional_indices(est, t)
        if isempty(idx)
            cross_sectional_blank_row!(Y, t)
            continue
        end
        v = sort!([A[t, i] for i in idx])
        qlo = Statistics.quantile(v, ct.low; sorted = true)
        qhi = Statistics.quantile(v, ct.high; sorted = true)
        for i in axes(A, 2)
            Y[t, i] = clamp(A[t, i], qlo, qhi)
        end
    end
    return Y
end
function cross_sectional_transform(ct::CrossSectionalTanhShrinker, X::MatNum;
                                   w::Option{<:MatNum} = nothing,
                                   groups::Option{<:AbstractMatrix{<:Integer}} = nothing)
    assert_cross_sectional_matrix(X)
    assert_cross_sectional_weights(X, w)
    assert_cross_sectional_groups(X, groups)
    T = float(promote_type(eltype(X), cross_sectional_weight_type(w), typeof(ct.knee),
                           typeof(ct.atol)))
    A = convert(Matrix{T}, X)
    fin = isfinite.(A)
    est = cross_sectional_estimation_mask(fin, w)
    Y = similar(A)
    for t in axes(A, 1)
        idx = cross_sectional_indices(est, t)
        if isempty(idx)
            cross_sectional_blank_row!(Y, t)
            continue
        end
        m = Statistics.median([A[t, i] for i in idx])
        s = CS_MAD_CONSISTENCY * Statistics.median([abs(A[t, i] - m) for i in idx])
        h = ct.knee * s
        for i in axes(A, 2)
            Y[t, i] = s > ct.atol ? m + h * tanh((A[t, i] - m) / h) : A[t, i]
        end
    end
    return Y
end
function cross_sectional_transform(ct::CrossSectionalStandardiser, X::MatNum;
                                   w::Option{<:MatNum} = nothing,
                                   groups::Option{<:AbstractMatrix{<:Integer}} = nothing)
    assert_cross_sectional_matrix(X)
    assert_cross_sectional_weights(X, w)
    assert_cross_sectional_groups(X, groups)
    T = float(promote_type(eltype(X), cross_sectional_weight_type(w), typeof(ct.atol)))
    A = convert(Matrix{T}, X)
    fin = isfinite.(A)
    est = cross_sectional_estimation_mask(fin, w)
    Y = similar(A)
    cross_sectional_standardise!(Y, A, fin, est, w, groups, ct.min_group_size, ct.atol)
    return Y
end
function cross_sectional_transform(ct::CrossSectionalGaussianRank, X::MatNum;
                                   w::Option{<:MatNum} = nothing,
                                   groups::Option{<:AbstractMatrix{<:Integer}} = nothing)
    assert_cross_sectional_matrix(X)
    assert_cross_sectional_weights(X, w)
    assert_cross_sectional_groups(X, groups)
    T = float(promote_type(eltype(X), cross_sectional_weight_type(w), typeof(ct.atol)))
    A = convert(Matrix{T}, X)
    fin = isfinite.(A)
    est = cross_sectional_estimation_mask(fin, w)
    P = cross_sectional_percentile_ranks(A, fin, est, groups, ct.min_group_size)
    Y = [isnan(p) ? p : sqrt(2 * one(p)) * SpecialFunctions.erfinv(2 * p - one(p))
         for p in P]
    cross_sectional_recentre_rescale!(Y, fin, est, w, ct.atol, ct.scale)
    return Y
end
function cross_sectional_transform(ct::CrossSectionalPercentileRank, X::MatNum;
                                   w::Option{<:MatNum} = nothing,
                                   groups::Option{<:AbstractMatrix{<:Integer}} = nothing)
    assert_cross_sectional_matrix(X)
    assert_cross_sectional_weights(X, w)
    assert_cross_sectional_groups(X, groups)
    T = float(promote_type(eltype(X), cross_sectional_weight_type(w)))
    A = convert(Matrix{T}, X)
    fin = isfinite.(A)
    est = cross_sectional_estimation_mask(fin, w)
    return cross_sectional_percentile_ranks(A, fin, est, groups, ct.min_group_size)
end
"""
    cross_sectional_standardise!(Y::AbstractMatrix, A::AbstractMatrix, fin::AbstractMatrix{Bool}, est::AbstractMatrix{Bool}, w::Option{<:MatNum}, groups::Nothing, mgs::Integer, atol::Real) -> nothing
    cross_sectional_standardise!(Y::AbstractMatrix, A::AbstractMatrix, fin::AbstractMatrix{Bool}, est::AbstractMatrix{Bool}, w::Option{<:MatNum}, groups::AbstractMatrix{<:Integer}, mgs::Integer, atol::Real) -> nothing

Score a matrix as cross-sectional z-scores, with or without groups.

Without group labels one centre and one scale serve the whole observation. With them each group that stands carries its own pair, every other asset takes the observation's pair, and the scores are recentred and rescaled over the whole observation afterwards.

# Algorithm

 1. Walk the observations, and take the estimation set of each.
 2. Compute the observation's own centre and scale.
 3. Without group labels, score every finite cell against that pair and stop.
 4. With them, compute the pair of each group that stands, give every other finite cell the observation's pair, score, and then recentre and rescale over the whole observation.

# Arguments

  - `Y::AbstractMatrix`: Output matrix `observations × assets`.
  - `A::AbstractMatrix`: Data matrix `observations × assets`, already floating point.
  - `fin::AbstractMatrix{Bool}`: Finiteness mask `observations × assets`.
  - `est::AbstractMatrix{Bool}`: Estimation mask `observations × assets`.
  - `w`: Benchmark weight matrix `observations × assets`, or `nothing`.
  - `groups`: Group label matrix `observations × assets`, or `nothing`.
  - `mgs::Integer`: $(arg_dict[:min_group_size])
  - `atol::Real`: $(arg_dict[:atol_cs])

# Returns

  - `nothing`.

# Related

  - [`CrossSectionalStandardiser`](@ref)
  - [`cross_sectional_zscore_row!`](@ref)
  - [`cross_sectional_recentre_rescale!`](@ref)
"""
function cross_sectional_standardise!(Y::AbstractMatrix, A::AbstractMatrix,
                                      fin::AbstractMatrix{Bool}, est::AbstractMatrix{Bool},
                                      w::Option{<:MatNum}, ::Nothing, ::Integer,
                                      atol::Real)::Nothing
    for t in axes(A, 1)
        idx = cross_sectional_indices(est, t)
        mu = isempty(idx) ? zero(eltype(A)) : cross_sectional_weighted_mean(A, w, t, idx)
        s = cross_sectional_equal_std(A, t, idx, mu)
        cross_sectional_zscore_row!(Y, A, fin, t, idx, mu, s, atol)
    end
    return nothing
end
function cross_sectional_standardise!(Y::AbstractMatrix, A::AbstractMatrix,
                                      fin::AbstractMatrix{Bool}, est::AbstractMatrix{Bool},
                                      w::Option{<:MatNum},
                                      groups::AbstractMatrix{<:Integer}, mgs::Integer,
                                      atol::Real)::Nothing
    for t in axes(A, 1)
        idx = cross_sectional_indices(est, t)
        mu = isempty(idx) ? zero(eltype(A)) : cross_sectional_weighted_mean(A, w, t, idx)
        s = cross_sectional_equal_std(A, t, idx, mu)
        gidx = cross_sectional_row_groups(est, groups, t)
        M, S = cross_sectional_cell_stats(A, w, t, groups, gidx, mgs, mu, s)
        cross_sectional_zscore_row!(Y, A, fin, t, idx, M, S, atol)
    end
    cross_sectional_recentre_rescale!(Y, fin, est, w, atol, true)
    return nothing
end
"""
    cross_sectional_cell_stats(A::AbstractMatrix, w::Option{<:MatNum}, t::Integer, groups::AbstractMatrix{<:Integer}, gidx::Dict{Int, Vector{Int}}, mgs::Integer, mu, sigma)

Return the centre and the scale that apply to each asset of one grouped observation.

An asset whose group stands takes its group's pair, and every other asset takes the observation's own pair. Building both vectors first keeps the scoring loop free of the fallback rule.

# Algorithm

 1. Compute the pair of each group holding at least `mgs` estimation assets.
 2. Fill both vectors with the observation's pair.
 3. Overwrite the entries of every asset whose group stands.

# Arguments

  - `A::AbstractMatrix`: Data matrix `observations × assets`, already floating point.
  - `w`: Benchmark weight matrix `observations × assets`, or `nothing`.
  - `t::Integer`: Observation.
  - `groups::AbstractMatrix{<:Integer}`: Group label matrix `observations × assets`.
  - `gidx::Dict{Int, Vector{Int}}`: Estimation set of each group of the observation.
  - `mgs::Integer`: $(arg_dict[:min_group_size])
  - `mu`: Centre of the whole observation.
  - `sigma`: Scale of the whole observation.

# Returns

  - `M::Vector{<:Number}`: Centre of each asset.
  - `S::Vector{<:Number}`: Scale of each asset.

# Related

  - [`cross_sectional_standardise!`](@ref)
  - [`cross_sectional_row_groups`](@ref)
"""
function cross_sectional_cell_stats(A::AbstractMatrix, w::Option{<:MatNum}, t::Integer,
                                    groups::AbstractMatrix{<:Integer},
                                    gidx::Dict{Int, Vector{Int}}, mgs::Integer, mu, sigma)
    st = Dict{Int, Tuple{eltype(A), eltype(A)}}()
    for (g, gi) in gidx
        if length(gi) >= mgs
            m = cross_sectional_weighted_mean(A, w, t, gi)
            st[g] = (m, cross_sectional_equal_std(A, t, gi, m))
        end
    end
    M = fill(mu, size(A, 2))
    S = fill(sigma, size(A, 2))
    for i in axes(A, 2)
        p = get(st, Int(groups[t, i]), nothing)
        if !isnothing(p)
            M[i] = p[1]
            S[i] = p[2]
        end
    end
    return M, S
end
"""
    cross_sectional_groups(B::AbstractArray{<:Real, 3}) -> Matrix{Int}
    cross_sectional_groups(pnl::AssetPanel, name::AbstractString) -> Matrix{Int}

Derive the group labels of a cross-sectional transform from a one-hot block of a Panel Field.

A categorical Panel Field claims one column of the feature axis per level, and an asset carries a one in the column of the level it belongs to. This turns that block into the `observations × assets` label matrix [`cross_sectional_transform`](@ref) takes, and it labels an asset whose row sets no level with [`CS_MISSING_GROUP`](@ref).

# Algorithm

 1. Label every asset [`CS_MISSING_GROUP`](@ref).
 2. Walk the levels of each asset, and label the asset with the first level it sets.

# Arguments

  - `B::AbstractArray{<:Real, 3}`: One-hot block `observations × assets × levels`.
  - `pnl::AssetPanel`: Asset Panel holding the categorical Panel Field.
  - `name::AbstractString`: Name of the categorical Panel Field to read.

# Validation

  - The named Panel Field is a [`CategoricalPanelField`](@ref). Raises an `ArgumentError`.

# Returns

  - `groups::Matrix{Int}`: Group label matrix `observations × assets`. A label is the position of the level in the Panel Field's own level order, and [`CS_MISSING_GROUP`](@ref) marks an asset that sets none.

# Examples

```jldoctest
julia> B = reshape([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 1, 2, 3)
1×2×3 Array{Float64, 3}:
[:, :, 1] =
 1.0  0.0

[:, :, 2] =
 0.0  1.0

[:, :, 3] =
 0.0  0.0

julia> cross_sectional_groups(B)
1×2 Matrix{Int64}:
 1  2
```

# Related

  - [`cross_sectional_transform`](@ref)
  - [`CS_MISSING_GROUP`](@ref)
  - [`AssetPanel`](@ref)
  - [`CategoricalPanelField`](@ref)
  - [`panel_field`](@ref)
"""
function cross_sectional_groups(B::AbstractArray{<:Real, 3})::Matrix{Int}
    G = fill(CS_MISSING_GROUP, size(B, 1), size(B, 2))
    for t in axes(B, 1), i in axes(B, 2)
        for l in axes(B, 3)
            if !iszero(B[t, i, l])
                G[t, i] = l
                break
            end
        end
    end
    return G
end
function cross_sectional_groups(pnl::AssetPanel, name::AbstractString)::Matrix{Int}
    f = panel_field(pnl, name)
    @argcheck(isa(f, CategoricalPanelField),
              ArgumentError("a group label is the code of a categorical Panel Field, so \"$name\" must be a CategoricalPanelField, got a $(nameof(typeof(f)))"))
    @argcheck(ndims(f.codes) == 2,
              DimensionMismatch("a group label is read per observation and asset, so the Panel Field \"$name\" must be time-varying; this Asset Panel is static"))
    return Matrix{Int}(f.codes)
end

export CrossSectionalWinsoriser, CrossSectionalTanhShrinker, CrossSectionalStandardiser,
       CrossSectionalGaussianRank, CrossSectionalPercentileRank, cross_sectional_transform,
       cross_sectional_groups
