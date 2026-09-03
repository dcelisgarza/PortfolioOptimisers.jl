"""
$(DocStringExtensions.TYPEDEF)

Partial-fit state of a [`Coskewness`](@ref) estimator under [`FullMoment`](@ref).

The state carries the running count, the running mean and the running second and third central co-moment accumulators. The third accumulator is the answer the estimator reports, and the second one is there because the update of the third reads it: a new observation moves the centre, and every past third co-moment is corrected by a term in the second one.

This type is an implementation detail and is not intended for direct use. [`partial_fit!`](@ref) writes it, [`coskewness`](@ref) reads it, and [`merge_states`](@ref) folds two of them.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`Coskewness`](@ref)
  - [`AbstractPartialFitState`](@ref)
  - [`CokurtosisPartialFitState`](@ref)
  - [`partial_fit!`](@ref)
  - [`merge_states`](@ref)
"""
@concrete struct CoskewnessPartialFitState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    $(field_dict[:pf_mu])
    """
    mu
    """
    $(field_dict[:pf_M2])
    """
    M2
    """
    $(field_dict[:pf_M3])
    """
    M3
end
"""
$(DocStringExtensions.TYPEDEF)

Partial-fit state of a [`Cokurtosis`](@ref) estimator under [`FullMoment`](@ref).

The state carries the running count, the running mean and the running second, third and fourth central co-moment accumulators. The fourth accumulator is the answer the estimator reports. The second and the third are there because the update of the fourth reads both: a new observation moves the centre, and every past fourth co-moment is corrected by terms in the third and in the second.

`M4` is `assets² × assets²`, so the state is the largest object in the incremental seam. At 100 assets it holds 10⁸ entries, which is 800 MB in double precision. The seam trades that memory for the recomputation it removes, and a caller who cannot pay it runs the batch verb instead.

This type is an implementation detail and is not intended for direct use. [`partial_fit!`](@ref) writes it, [`cokurtosis`](@ref) reads it, and [`merge_states`](@ref) folds two of them.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`Cokurtosis`](@ref)
  - [`AbstractPartialFitState`](@ref)
  - [`CoskewnessPartialFitState`](@ref)
  - [`partial_fit!`](@ref)
  - [`merge_states`](@ref)
"""
@concrete struct CokurtosisPartialFitState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    $(field_dict[:pf_mu])
    """
    mu
    """
    $(field_dict[:pf_M2])
    """
    M2
    """
    $(field_dict[:pf_M3])
    """
    M3
    """
    `M4`: Running fourth central co-moment accumulator, `assets² × assets²`. It is the sum over the observations and not the cokurtosis, so a read-out divides it by `n`.
    """
    M4
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the count, the mean and the central co-moment accumulators of one block of observations.

The accumulators are the numerators of the batch verbs: `M2` divided by `n` is the covariance the batch route computes, `M3` divided by `n` is the coskewness tensor, and `transpose(z) * z` divided by `n` is the cokurtosis matrix. The block of one observation has a mean equal to that observation and three accumulators of zero, which is what makes one `partial_fit!` per row the Welford update.

A pair of assets ``(p, q)`` is one column of the third accumulator and one row or column of the fourth, at position ``(p - 1) N + q``, with `p` the slow index and `q` the fast one. That is the ordering `kron(o, Y) ⊙ kron(Y, o)` produces, so nothing is transposed between the incremental route and the batch one.

# Algorithm

 1. Take the mean of `X` over its observations, giving `mu`.
 2. Subtract `mu` from every observation, giving the deviation matrix `Y`.
 3. Form `z`, whose column ``(p - 1) N + q`` is the elementwise product of columns `p` and `q` of `Y`.
 4. Return `n`, `mu`, `transpose(Y) * Y`, `transpose(Y) * z` and `z`.

# Arguments

  - `X`: Block of observations (observations × assets).

# Returns

  - `(n, mu, M2, M3, z)::Tuple`: The count, the mean, the second and third accumulators, and the pairwise deviation product `z`, which the fourth accumulator squares.

# Related

  - [`shift_comoment3`](@ref)
  - [`shift_comoment4`](@ref)
  - [`partial_fit!`](@ref)
"""
function comoment_block(X::MatNum)
    n = size(X, 1)
    mu = vec(Statistics.mean(X; dims = 1))
    Y = X .- transpose(mu)
    o = transpose(range(one(eltype(Y)), one(eltype(Y)); length = size(Y, 2)))
    z = kron(o, Y) ⊙ kron(Y, o)
    return n, mu, transpose(Y) * Y, transpose(Y) * z, z
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Moves the third central co-moment accumulator of one block from its own mean to a shifted centre.

This is the whole mathematics of the higher-moment incremental fit, and both [`partial_fit!`](@ref) and [`merge_states`](@ref) reach it. A block records its co-moments about its own mean. Two blocks folded into one have a common mean that is neither of theirs, so each block's accumulator is first moved to the common centre and the two are then added. The one-observation update is the same operation with a block of one row, whose accumulators are zero.

# Mathematical definition

Write ``y_{ti} = x_{ti} - \\mu_i`` for the deviations of the block about its own mean, and let the centre move to ``\\mu + a``. The deviations become ``y_{ti} - a_i``, and the sum of the deviations about the block's own mean is zero. Expanding the product of three shifted deviations and dropping every term that carries a lone ``\\sum_t y_t`` gives

```math
\\begin{align}
M_{3, kpq}(\\mu + a) &= M_{3, kpq} - a_k M_{2, pq} - a_p M_{2, kq} - a_q M_{2, kp} - m\\, a_k a_p a_q\\, .
\\end{align}
```

Where:

  - ``m``: number of observations in the block.
  - ``a``: displacement of the centre.
  - ``M_2``, ``M_3``: second and third central co-moment accumulators of the block about its own mean.

The three correction terms read the **old** accumulator ``M_2``, so a caller shifts the third accumulator before it overwrites the second one. The scalar case ``N = 1`` is the univariate Welford recursion ``M_3 \\leftarrow M_3 - 3 a M_2 - m a^3``.

# Algorithm

 1. Form the pairwise product `az` of the displacement with itself, ordered as in [`comoment_block`](@ref).
 2. Take the three correction terms in the second accumulator, one for each index of the third.
 3. Subtract them, and subtract the displacement cubed scaled by the block count.

# Arguments

  - `m`: Number of observations in the block.
  - `a`: Displacement of the centre, `assets × 1`.
  - `M2`: Second central co-moment accumulator of the block, about its own mean.
  - `M3`: Third central co-moment accumulator of the block, about its own mean.

# Returns

  - `M3::MatNum`: The third accumulator of the same block about the shifted centre.

# Related

  - [`shift_comoment4`](@ref)
  - [`comoment_block`](@ref)
  - [`merge_states`](@ref)
"""
function shift_comoment3(m::Number, a::VecNum, M2::MatNum, M3::MatNum)
    at = transpose(a)
    az = kron(a, a)
    return M3 .- a * transpose(vec(M2)) .- kron(at, M2) .- kron(M2, at) .-
           m .* (a * transpose(az))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Moves the fourth central co-moment accumulator of one block from its own mean to a shifted centre.

The companion of [`shift_comoment3`](@ref), derived from the same expansion carried one order further. It reads the third and the second accumulator of the block, so a caller shifts the fourth one first.

# Mathematical definition

With the notation of [`shift_comoment3`](@ref), expanding the product of four shifted deviations and dropping every term that carries a lone ``\\sum_t y_t`` gives

```math
\\begin{align}
M_{4, pqrs}(\\mu + a) &= M_{4, pqrs}
 - \\left(a_p M_{3, qrs} + a_q M_{3, prs} + a_r M_{3, spq} + a_s M_{3, rpq}\\right)\\\\
 &+ \\left(a_p a_q M_{2, rs} + a_p a_r M_{2, qs} + a_p a_s M_{2, qr}
 + a_q a_r M_{2, ps} + a_q a_s M_{2, pr} + a_r a_s M_{2, pq}\\right)\\\\
 &+ m\\, a_p a_q a_r a_s\\, .
\\end{align}
```

Where:

  - ``m``: number of observations in the block.
  - ``a``: displacement of the centre.
  - ``M_2``, ``M_3``, ``M_4``: second, third and fourth central co-moment accumulators of the block about its own mean.

The four third-order terms are one per index, and the six second-order terms are one per pair of indices. Both accumulators are fully symmetric in their indices, so each term is one Kronecker product of the displacement with the accumulator. The scalar case ``N = 1`` is the univariate Welford recursion ``M_4 \\leftarrow M_4 - 4 a M_3 + 6 a^2 M_2 + m a^4``.

# Algorithm

 1. Form the pairwise product `az` of the displacement with itself, and the outer product `aa`.
 2. Subtract the four third-order terms, one for each index of the fourth accumulator.
 3. Add the six second-order terms, one for each pair of indices.
 4. Add the displacement to the fourth power, scaled by the block count.

# Arguments

  - `m`: Number of observations in the block.
  - `a`: Displacement of the centre, `assets × 1`.
  - `M2`: Second central co-moment accumulator of the block, about its own mean.
  - `M3`: Third central co-moment accumulator of the block, about its own mean.
  - `M4`: Fourth central co-moment accumulator of the block, about its own mean.

# Returns

  - `M4::MatNum`: The fourth accumulator of the same block about the shifted centre.

# Related

  - [`shift_comoment3`](@ref)
  - [`comoment_block`](@ref)
  - [`merge_states`](@ref)
"""
function shift_comoment4(m::Number, a::VecNum, M2::MatNum, M3::MatNum, M4::MatNum)
    at = transpose(a)
    aa = a * at
    az = kron(a, a)
    v2 = vec(M2)
    M3t = transpose(M3)
    cross = kron(a, kron(M2, at))
    return M4 .- kron(a, M3) .- kron(M3, a) .- kron(at, M3t) .- kron(M3t, at) .+
           az * transpose(v2) .+ v2 * transpose(az) .+ kron(aa, M2) .+ kron(M2, aa) .+
           cross .+ transpose(cross) .+ m .* (az * transpose(az))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an estimator whose configuration no incremental fit reproduces, naming the field that refuses it.

A partial fit keeps the running sample mean and the central co-moments about it, and nothing else of the sample. Two configuration choices break that, and both are refused rather than approximated.

  - **Observation weights.** A weights vector describes a sample of a fixed length, and a [`DynamicAbstractWeights`](@ref) derives every weight from the length of the sample. Either way a new observation reweights every past one, so no state written before it is still valid.
  - **A mean estimator that is not the plain sample mean.** The state carries the running sample mean, so an estimator whose centre is a shrunk, a median or a custom value would be centred on a quantity the state does not hold.

An estimator that carries no centring field hands over `nothing`, which the `Nothing` method reads as [`SimpleExpectedReturns`](@ref), because that is the centre such an estimator takes.

# Arguments

  - `me`: Expected returns estimator of the estimator being fitted.
  - `w`: Observation weights of the estimator being fitted.
  - `name`: Name of the estimator, used in the message.

# Validation

  - `w` is `nothing`. An `ArgumentError` is thrown otherwise.
  - `me` is a [`SimpleExpectedReturns`](@ref) whose own `w` is `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`partial_fit!`](@ref)
  - [`SimpleExpectedReturns`](@ref)
"""
function assert_partial_fittable(me::AbstractExpectedReturnsEstimator,
                                 w::Option{<:ObsWeights}, name::AbstractString)
    @argcheck(isnothing(w),
              ArgumentError("$name cannot be fitted incrementally while `w` is set, because observation weights reweight every past observation when a new one arrives. Set `w = nothing`, or run the batch verb."))
    @argcheck(isa(me, SimpleExpectedReturns) && isnothing(me.w),
              ArgumentError("$name cannot be fitted incrementally with `me = $(typeof(me))`, because a partial-fit state carries the running sample mean and no other centre. Set `me = SimpleExpectedReturns()`, or run the batch verb."))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`Nothing` method of [`assert_partial_fittable`](@ref). An estimator that carries no centring field, and one whose `nothing` selects [`SimpleExpectedReturns`](@ref), both centre on the plain sample mean, so the pair is checked against that estimator and only the weights can refuse it.
"""
function assert_partial_fittable(::Nothing, w::Option{<:ObsWeights}, name::AbstractString)
    return assert_partial_fittable(SimpleExpectedReturns(), w, name)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`CoskewnessPartialFitState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

The count, the mean and the second accumulator take Chan's merge. The third accumulator is moved to the common mean one block at a time, with [`shift_comoment3`](@ref), and the two shifted accumulators are added.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref).
 2. Fold the count, the mean and the second accumulator with [`chan_merge`](@ref).
 3. Take the displacement of each block's own mean to the common mean.
 4. Shift each block's third accumulator to the common mean, and add the two.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - `state::CoskewnessPartialFitState`: The state the two blocks give when they are fitted as one block.

# Related

  - [`CoskewnessPartialFitState`](@ref)
  - [`shift_comoment3`](@ref)
  - [`chan_merge`](@ref)
"""
function merge_states(a::CoskewnessPartialFitState, b::CoskewnessPartialFitState)
    assert_mergeable_states(a, b)
    n, mu, M2 = chan_merge(a.n, a.mu, a.M2, b.n, b.mu, b.M2)
    M3 = shift_comoment3(a.n, mu .- a.mu, a.M2, a.M3) .+
         shift_comoment3(b.n, mu .- b.mu, b.M2, b.M3)
    return CoskewnessPartialFitState(n, mu, M2, M3)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`CokurtosisPartialFitState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

The companion of the [`CoskewnessPartialFitState`](@ref) merge, carried one order further. The fourth accumulator is shifted before the third, because [`shift_comoment4`](@ref) reads the third accumulator of the block about its own mean.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref).
 2. Fold the count, the mean and the second accumulator with [`chan_merge`](@ref).
 3. Take the displacement of each block's own mean to the common mean.
 4. Shift each block's fourth accumulator to the common mean, and add the two.
 5. Shift each block's third accumulator to the common mean, and add the two.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - `state::CokurtosisPartialFitState`: The state the two blocks give when they are fitted as one block.

# Related

  - [`CokurtosisPartialFitState`](@ref)
  - [`shift_comoment4`](@ref)
  - [`chan_merge`](@ref)
"""
function merge_states(a::CokurtosisPartialFitState, b::CokurtosisPartialFitState)
    assert_mergeable_states(a, b)
    n, mu, M2 = chan_merge(a.n, a.mu, a.M2, b.n, b.mu, b.M2)
    da = mu .- a.mu
    db = mu .- b.mu
    M4 = shift_comoment4(a.n, da, a.M2, a.M3, a.M4) .+
         shift_comoment4(b.n, db, b.M2, b.M3, b.M4)
    M3 = shift_comoment3(a.n, da, a.M2, a.M3) .+ shift_comoment3(b.n, db, b.M2, b.M3)
    return CokurtosisPartialFitState(n, mu, M2, M3, M4)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds a block of observations into the partial-fit state of a [`Coskewness`](@ref) estimator under [`FullMoment`](@ref).

The estimator that carries no state adopts the block's state. The estimator that carries one merges the two, so the state after any sequence of blocks is the state of the concatenated sample. One row per call is the Welford update, and it is exact: the block of one observation carries zero accumulators, and the whole correction is the shift of the running state to the new mean.

The method builds a new state rather than writing into the arrays of the old one, because the merge of two blocks is written once and shared with [`merge_states`](@ref). The returned estimator therefore holds a state that does **not** alias the one it was built from, and an estimator a caller kept from an earlier call still reads the sample it was shown. That is narrower than the general [`partial_fit!`](@ref) contract, and it is the reading issue #712 asks the maintainer to settle for the whole seam.

# Algorithm

 1. Refuse the configuration no incremental fit reproduces, with [`assert_partial_fittable`](@ref).
 2. Orient `X` to observations × assets with [`dims_oriented`](@ref), which validates `dims`.
 3. Build the state of the block alone, with [`comoment_block`](@ref).
 4. Merge it into the state the estimator carries, unless the estimator carries none.
 5. Rebind `cache` with `Accessors.@reset`, and return the estimator.

# Arguments

  - `ske`: Coskewness estimator with a [`FullMoment`](@ref) moment algorithm.
  - `X`: Block of observations (observations × assets).
  - $(arg_dict[:dims])

# Validation

  - `ske.w` is `nothing` and `ske.me` is an unweighted [`SimpleExpectedReturns`](@ref).
  - `dims` is either `1` or `2`.
  - A state already held has the asset count of `X`.

# Returns

  - `ske::Coskewness`: The estimator carrying the folded state.

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = randn(rng, 10, 3);

julia> ske = Coskewness();

julia> for i in axes(X, 1)
           global ske = partial_fit!(ske, X[i:i, :])
       end

julia> cskew, V = coskewness(ske);

julia> isapprox(cskew, first(coskewness(Coskewness(), X)))
true
```

# Related

  - [`Coskewness`](@ref)
  - [`CoskewnessPartialFitState`](@ref)
  - [`coskewness`](@ref)
  - [`merge_states`](@ref)
"""
function partial_fit!(ske::Coskewness{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1)
    assert_partial_fittable(ske.me, ske.w, "Coskewness")
    X = dims_oriented(dims, X)
    n, mu, M2, M3, _ = comoment_block(X)
    state = CoskewnessPartialFitState(n, mu, M2, M3)
    if !isnothing(ske.cache)
        state = merge_states(ske.cache, state)
    end
    Accessors.@reset ske.cache = state
    return ske
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds one observation into the partial-fit state of a [`Coskewness`](@ref) estimator under [`FullMoment`](@ref).

The single-observation arm of the [`partial_fit!`](@ref) interface. The entries of `x` are the assets, so the observation is the block of one row.

# Algorithm

 1. Reshape `x` to a block of one row.
 2. Forward to the block method of [`partial_fit!`](@ref).

# Arguments

  - `ske`: Coskewness estimator with a [`FullMoment`](@ref) moment algorithm.
  - `x`: One observation, whose entries are the assets.

# Returns

  - `ske::Coskewness`: The estimator carrying the folded state.

# Related

  - [`Coskewness`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(ske::Coskewness{<:Any, <:Any, <:FullMoment}, x::VecNum)
    return partial_fit!(ske, reshape(x, 1, length(x)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an incremental fit of a [`Coskewness`](@ref) estimator under [`SemiMoment`](@ref).

[`SemiMoment`](@ref) clips every positive deviation to zero before the moment, and the centre it clips against is a statistic of the whole sample. A new observation moves that centre, so every past clip moves with it and a past observation's membership of the clipped set flips. No state written before that observation is still valid, which is the second clause of the membership rule of issue #308.

# Arguments

  - `ske`: Coskewness estimator with a [`SemiMoment`](@ref) moment algorithm.
  - `args...`: Ignored.
  - `kwargs...`: Ignored.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`Coskewness`](@ref)
  - [`SemiMoment`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(::Coskewness{<:Any, <:Any, <:SemiMoment}, args...; kwargs...)
    return throw(ArgumentError("a `Coskewness` estimator with a `SemiMoment` moment algorithm cannot be fitted incrementally, because the clip against the sample mean moves when the mean moves, so a past observation's membership of the clipped set flips. Use a `FullMoment` algorithm, or run the batch verb."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds a block of observations into the partial-fit state of a [`Cokurtosis`](@ref) estimator under [`FullMoment`](@ref).

The companion of the [`Coskewness`](@ref) method, carried one order further. The state it writes is `assets² × assets²` in its largest field, so the memory the seam trades for the recomputation it removes is stated on [`CokurtosisPartialFitState`](@ref). It builds a new state rather than writing into the arrays of the old one, for the reason the [`Coskewness`](@ref) method states, and issue #712 asks the maintainer to settle that reading for the whole seam.

# Algorithm

 1. Refuse the configuration no incremental fit reproduces, with [`assert_partial_fittable`](@ref).
 2. Orient `X` to observations × assets with [`dims_oriented`](@ref), which validates `dims`.
 3. Build the state of the block alone, with [`comoment_block`](@ref), and square its pairwise deviation product for the fourth accumulator.
 4. Merge it into the state the estimator carries, unless the estimator carries none.
 5. Rebind `cache` with `Accessors.@reset`, and return the estimator.

# Arguments

  - `kte`: Cokurtosis estimator with a [`FullMoment`](@ref) moment algorithm.
  - `X`: Block of observations (observations × assets).
  - $(arg_dict[:dims])

# Validation

  - `kte.w` is `nothing` and `kte.me` is an unweighted [`SimpleExpectedReturns`](@ref).
  - `dims` is either `1` or `2`.
  - A state already held has the asset count of `X`.

# Returns

  - `kte::Cokurtosis`: The estimator carrying the folded state.

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = randn(rng, 10, 2);

julia> kte = Cokurtosis();

julia> for i in axes(X, 1)
           global kte = partial_fit!(kte, X[i:i, :])
       end

julia> isapprox(cokurtosis(kte), cokurtosis(Cokurtosis(), X))
true
```

# Related

  - [`Cokurtosis`](@ref)
  - [`CokurtosisPartialFitState`](@ref)
  - [`cokurtosis`](@ref)
  - [`merge_states`](@ref)
"""
function partial_fit!(kte::Cokurtosis{<:Any, <:Any, <:FullMoment}, X::MatNum; dims::Int = 1)
    assert_partial_fittable(kte.me, kte.w, "Cokurtosis")
    X = dims_oriented(dims, X)
    n, mu, M2, M3, z = comoment_block(X)
    state = CokurtosisPartialFitState(n, mu, M2, M3, transpose(z) * z)
    if !isnothing(kte.cache)
        state = merge_states(kte.cache, state)
    end
    Accessors.@reset kte.cache = state
    return kte
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds one observation into the partial-fit state of a [`Cokurtosis`](@ref) estimator under [`FullMoment`](@ref).

The single-observation arm of the [`partial_fit!`](@ref) interface. The entries of `x` are the assets, so the observation is the block of one row.

# Algorithm

 1. Reshape `x` to a block of one row.
 2. Forward to the block method of [`partial_fit!`](@ref).

# Arguments

  - `kte`: Cokurtosis estimator with a [`FullMoment`](@ref) moment algorithm.
  - `x`: One observation, whose entries are the assets.

# Returns

  - `kte::Cokurtosis`: The estimator carrying the folded state.

# Related

  - [`Cokurtosis`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(kte::Cokurtosis{<:Any, <:Any, <:FullMoment}, x::VecNum)
    return partial_fit!(kte, reshape(x, 1, length(x)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an incremental fit of a [`Cokurtosis`](@ref) estimator under [`SemiMoment`](@ref).

The same refusal as the [`Coskewness`](@ref) one, for the same reason: [`SemiMoment`](@ref) clips against a centre that a new observation moves, so a past observation's membership of the clipped set flips.

# Arguments

  - `kte`: Cokurtosis estimator with a [`SemiMoment`](@ref) moment algorithm.
  - `args...`: Ignored.
  - `kwargs...`: Ignored.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`Cokurtosis`](@ref)
  - [`SemiMoment`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(::Cokurtosis{<:Any, <:Any, <:SemiMoment}, args...; kwargs...)
    return throw(ArgumentError("a `Cokurtosis` estimator with a `SemiMoment` moment algorithm cannot be fitted incrementally, because the clip against the sample mean moves when the mean moves, so a past observation's membership of the clipped set flips. Use a `FullMoment` algorithm, or run the batch verb."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the coskewness tensor and its negative spectral skewness matrix out of a partial-fit state.

The state read-out of [`coskewness`](@ref). It returns the pair the batch method returns, to machine precision, for the sample the state was fitted on.

Only the shape of that sample survives a partial fit, so the matrix processing estimator is handed a matrix of the right shape whose entries are zero. Of the shipped steps only `:dn` reads it, and it reads only `size(X)`, so the two routes agree. A custom `:alg` step that reads the values of the sample must be run from the batch verb.

# Algorithm

 1. Divide the third accumulator by the observation count, giving the coskewness tensor.
 2. Reduce it with [`negative_spectral_coskewness`](@ref), under `ske.mp` and the shape of the fitted sample.

# Arguments

  - `ske`: Coskewness estimator with a [`FullMoment`](@ref) moment algorithm.
  - `state`: Partial-fit state written by [`partial_fit!`](@ref).

# Returns

  - $(ret_dict[:cskew])
  - $(ret_dict[:cskewV])

# Related

  - [`Coskewness`](@ref)
  - [`CoskewnessPartialFitState`](@ref)
  - [`partial_fit!`](@ref)
  - [`negative_spectral_coskewness`](@ref)
"""
function coskewness(ske::Coskewness{<:Any, <:Any, <:FullMoment},
                    state::CoskewnessPartialFitState)
    cskew = state.M3 ./ state.n
    shape = SparseArrays.spzeros(eltype(cskew), state.n, length(state.mu))
    return cskew, negative_spectral_coskewness(cskew, shape, ske.mp)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the coskewness tensor and its negative spectral skewness matrix out of the state the estimator carries.

The one-argument forwarder of [`coskewness`](@ref). It reads `ske.cache`, which [`partial_fit!`](@ref) writes, and refuses an estimator that has been shown no observation.

# Algorithm

 1. Refuse the estimator whose `cache` is `nothing`.
 2. Forward to the state method of [`coskewness`](@ref).

# Arguments

  - `ske`: Coskewness estimator with a [`FullMoment`](@ref) moment algorithm.

# Validation

  - `ske.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:cskew])
  - $(ret_dict[:cskewV])

# Related

  - [`Coskewness`](@ref)
  - [`partial_fit!`](@ref)
"""
function coskewness(ske::Coskewness{<:Any, <:Any, <:FullMoment})
    @argcheck(!isnothing(ske.cache),
              ArgumentError("this `Coskewness` estimator carries no partial-fit state, so there is nothing to read out. Call `partial_fit!` first, or pass a data matrix."))
    return coskewness(ske, ske.cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the square cokurtosis matrix out of a partial-fit state.

The state read-out of [`cokurtosis`](@ref). It returns the matrix the batch method returns, to machine precision, for the sample the state was fitted on.

Only the shape of that sample survives a partial fit, so the matrix processing estimator is handed a matrix of the right shape whose entries are zero. Of the shipped steps only `:dn` reads it, and it reads only `size(X)`, so the two routes agree. A custom `:alg` step that reads the values of the sample must be run from the batch verb.

# Algorithm

 1. Divide the fourth accumulator by the observation count, giving the cokurtosis matrix.
 2. Process it in place with [`matrix_processing!`](@ref), under `kte.mp` and the shape of the fitted sample.

# Arguments

  - `kte`: Cokurtosis estimator with a [`FullMoment`](@ref) moment algorithm.
  - `state`: Partial-fit state written by [`partial_fit!`](@ref).

# Returns

  - $(ret_dict[:ckurt])

# Related

  - [`Cokurtosis`](@ref)
  - [`CokurtosisPartialFitState`](@ref)
  - [`partial_fit!`](@ref)
  - [`matrix_processing!`](@ref)
"""
function cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:FullMoment},
                    state::CokurtosisPartialFitState)
    ckurt = state.M4 ./ state.n
    shape = SparseArrays.spzeros(eltype(ckurt), state.n, length(state.mu))
    matrix_processing!(kte.mp, ckurt, shape)
    return ckurt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the square cokurtosis matrix out of the state the estimator carries.

The one-argument forwarder of [`cokurtosis`](@ref). It reads `kte.cache`, which [`partial_fit!`](@ref) writes, and refuses an estimator that has been shown no observation.

# Algorithm

 1. Refuse the estimator whose `cache` is `nothing`.
 2. Forward to the state method of [`cokurtosis`](@ref).

# Arguments

  - `kte`: Cokurtosis estimator with a [`FullMoment`](@ref) moment algorithm.

# Validation

  - `kte.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:ckurt])

# Related

  - [`Cokurtosis`](@ref)
  - [`partial_fit!`](@ref)
"""
function cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:FullMoment})
    @argcheck(!isnothing(kte.cache),
              ArgumentError("this `Cokurtosis` estimator carries no partial-fit state, so there is nothing to read out. Call `partial_fit!` first, or pass a data matrix."))
    return cokurtosis(kte, kte.cache)
end
