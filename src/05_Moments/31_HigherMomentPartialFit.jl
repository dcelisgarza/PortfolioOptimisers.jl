"""
$(DocStringExtensions.TYPEDEF)

Partial-fit state of a [`Coskewness`](@ref) estimator under [`FullMoment`](@ref).

The state carries the running count, the running mean and the running second and third central co-moment accumulators. The estimator reports the third accumulator. The state keeps the second one because the update of the third reads it. A new observation moves the centre, and the update corrects every past third co-moment by a term in the second one.

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

The state carries the running count, the running mean and the running second, third and fourth central co-moment accumulators. The estimator reports the fourth accumulator. The state keeps the second and the third because the update of the fourth reads both. A new observation moves the centre, and the update corrects every past fourth co-moment by terms in the third and in the second.

`M4` is `assets² × assets²`, so the state is the largest object of the incremental fit. At 100 assets it holds 10⁸ entries, which is 800 MB in double precision. The incremental fit pays that memory to read each observation once. A caller who cannot pay it runs the batch verb instead.

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
    `M4`: Running fourth central co-moment accumulator, `assets² × assets²`. It is the sum over the observations and not the cokurtosis, so `cokurtosis(kte)` with no data divides it by `n`.
    """
    M4
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Builds the count, the mean and the central co-moment accumulators of one block of observations.

The accumulators are the numerators of the batch verbs. `M2` divided by `n` is the covariance with the divisor `n`, `M3` divided by `n` is the coskewness tensor, and `transpose(z) * z` divided by `n` is the cokurtosis matrix. The block of one observation has a mean equal to that observation and three accumulators of zero. So one `partial_fit!` per row is the update of one observation, Corollary 2.2 of [pebay2008](@cite).

A pair of assets ``(p, q)`` is one column of the third accumulator and one row or column of the fourth, at position ``(p - 1) N + q``, with `p` the slow index and `q` the fast one. The batch verbs build `z` with the same `kron(o, Y) ⊙ kron(Y, o)`, so the incremental route and the batch route order the pairs in the same way.

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

  - [`shift_comoment3!`](@ref)
  - [`shift_comoment4!`](@ref)
  - [`partial_fit!`](@ref)

# References

  - $(ref_dict[:pebay2008]) Corollary 2.2, page 10.
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

[`partial_fit!`](@ref) and [`merge_states`](@ref) both reach this shift, and with [`shift_comoment4!`](@ref) it is the whole of the higher-moment merge. A block records its co-moments about its own mean. Two blocks folded into one have a common mean that differs from both, so the merge moves the accumulator of each block to the common centre and then adds the two. The update of one observation is the same operation with a block of one row, whose accumulators are zero.

# Mathematical definition

Write ``y_{ti} = x_{ti} - \\mu_i`` for the deviations of the block about its own mean, and let the centre move to ``\\mu + a``. The deviations become ``y_{ti} - a_i``, and the sum of the deviations about the block's own mean is zero. The expansion of the product of three shifted deviations, without each term that carries a lone ``\\sum_t y_t``, gives

```math
\\begin{align}
M_{3, kpq}(\\mu + a) &= M_{3, kpq} - a_k M_{2, pq} - a_p M_{2, kq} - a_q M_{2, kp} - m\\, a_k a_p a_q\\, .
\\end{align}
```

Where:

  - ``m``: number of observations in the block.
  - ``a``: displacement of the centre.
  - ``M_2``, ``M_3``: second and third central co-moment accumulators of the block about its own mean.

The three correction terms read the **old** accumulator ``M_2``, so a caller shifts the third accumulator before it overwrites the second one. The scalar case ``N = 1`` is ``M_3 \\leftarrow M_3 - 3 a M_2 - m a^3``. That is the term in the first block of Equation 2.6 of [pebay2008](@cite) at ``p = 3``, which states the shift for one series and any order. The expansion above applies the same step to each entry of the co-moment tensor.

The function adds the shifted accumulator into `out`, so a merge writes the two shifted blocks into one array. It takes each entry from `a`, `M2` and `M3` in one loop, and it allocates nothing.

# Algorithm

 1. Visit each entry ``(k, (p - 1) N + q)`` of `out`, ordered as in [`comoment_block`](@ref).
 2. Take the three correction terms in the second accumulator, one for each index of the third, and the displacement cubed scaled by the block count.
 3. Add the entry of `M3`, less those terms, to the entry of `out`.

# Arguments

  - `out`: The accumulator to add into, of the shape of `M3`. The function writes into it.
  - `m`: Number of observations in the block.
  - `a`: Displacement of the centre, `assets × 1`.
  - `M2`: Second central co-moment accumulator of the block, about its own mean.
  - `M3`: Third central co-moment accumulator of the block, about its own mean.

# Validation

  - With `N = length(a)`, `size(M2) == (N, N)`, and `size(M3) == size(out) == (N, N^2)`. Otherwise, the function raises a `DimensionMismatch`, before the loop reads an entry under `@inbounds`.
  - Every array has one-based axes. Otherwise, `Base.require_one_based_indexing` raises an `ArgumentError`.

# Returns

  - `out::MatNum`: `out` plus the third accumulator of the same block about the shifted centre.

# Related

  - [`shift_comoment4!`](@ref)
  - [`comoment_block`](@ref)
  - [`merge_states`](@ref)

# References

  - $(ref_dict[:pebay2008]) Equation 2.6, page 10. $(ref_dict[:no_original_source])
"""
function shift_comoment3!(out::MatNum, m::Number, a::VecNum, M2::MatNum, M3::MatNum)
    N = length(a)
    Base.require_one_based_indexing(out, a, M2, M3)
    @argcheck((size(M2), size(M3), size(out)) == ((N, N), (N, N^2), (N, N^2)),
              DimensionMismatch("`shift_comoment3!` needs `M2` of size $((N, N)), and `M3` and `out` of size $((N, N^2)), for the $N entries of `a`. Got\nsize(M2) => $(size(M2))\nsize(M3) => $(size(M3))\nsize(out) => $(size(out))."))
    @inbounds for p in 1:N, q in 1:N, k in 1:N
        c = (p - 1) * N + q
        out[k, c] += M3[k, c] - a[k] * M2[q, p] - a[p] * M2[k, q] - M2[k, p] * a[q] -
                     m * a[k] * a[p] * a[q]
    end
    return out
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Moves the fourth central co-moment accumulator of one block from its own mean to a shifted centre.

The companion of [`shift_comoment3!`](@ref), derived from the same expansion carried one order further. It reads the third and the second accumulator of the block, so a caller shifts the fourth one first.

# Mathematical definition

With the notation of [`shift_comoment3!`](@ref), the expansion of the product of four shifted deviations, without each term that carries a lone ``\\sum_t y_t``, gives

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

The four third-order terms are one per index, and the six second-order terms are one per pair of indices. The scalar case ``N = 1`` is ``M_4 \\leftarrow M_4 - 4 a M_3 + 6 a^2 M_2 + m a^4``, the term in the first block of Equation 2.6 of [pebay2008](@cite) at ``p = 4``. That equation states the shift for one series, and the expansion above applies the same step to each entry of the co-moment tensor.

The function adds the shifted accumulator into `out`, so a merge writes the two shifted blocks into one array of the size of `M4`. It takes each entry from `a`, `M2` and `M3` in one loop. Its one allocation is a transposed copy of `M3`, of `assets³` entries, so that the loop reads both forms of the third-order term down a column. At 30 assets, the copy makes the call 23 % faster. A form that builds each of the ten terms as an `assets² × assets²` array allocated 68 MiB per call.

# Algorithm

 1. Transpose `M3`. Then visit each entry ``((p - 1) N + q, (r - 1) N + s)`` of `out`, ordered as in [`comoment_block`](@ref).
 2. Take the four third-order terms, one for each index of the fourth accumulator.
 3. Take the six second-order terms, one for each pair of indices, and the displacement to the fourth power scaled by the block count.
 4. Add the entry of `M4`, less the third-order terms and plus the others, to the entry of `out`.

# Arguments

  - `out`: The accumulator to add into, of the shape of `M4`. The function writes into it.
  - `m`: Number of observations in the block.
  - `a`: Displacement of the centre, `assets × 1`.
  - `M2`: Second central co-moment accumulator of the block, about its own mean.
  - `M3`: Third central co-moment accumulator of the block, about its own mean.
  - `M4`: Fourth central co-moment accumulator of the block, about its own mean.

# Validation

  - With `N = length(a)`, `size(M2) == (N, N)`, `size(M3) == (N, N^2)`, and `size(M4) == size(out) == (N^2, N^2)`. Otherwise, the function raises a `DimensionMismatch`, before the loop reads an entry under `@inbounds`.
  - Every array has one-based axes. Otherwise, `Base.require_one_based_indexing` raises an `ArgumentError`.

# Returns

  - `out::MatNum`: `out` plus the fourth accumulator of the same block about the shifted centre.

# Related

  - [`shift_comoment3!`](@ref)
  - [`comoment_block`](@ref)
  - [`merge_states`](@ref)

# References

  - $(ref_dict[:pebay2008]) Equation 2.6, page 10. $(ref_dict[:no_original_source])
"""
function shift_comoment4!(out::MatNum, m::Number, a::VecNum, M2::MatNum, M3::MatNum,
                          M4::MatNum)
    N = length(a)
    Base.require_one_based_indexing(out, a, M2, M3, M4)
    @argcheck((size(M2), size(M3), size(M4), size(out)) ==
              ((N, N), (N, N^2), (N^2, N^2), (N^2, N^2)),
              DimensionMismatch("`shift_comoment4!` needs `M2` of size $((N, N)), `M3` of size $((N, N^2)), and `M4` and `out` of size $((N^2, N^2)), for the $N entries of `a`. Got\nsize(M2) => $(size(M2))\nsize(M3) => $(size(M3))\nsize(M4) => $(size(M4))\nsize(out) => $(size(out))."))
    M3t = permutedims(M3)
    @inbounds for r in 1:N, s in 1:N, p in 1:N, q in 1:N
        i = (p - 1) * N + q
        j = (r - 1) * N + s
        ap, aq, ar, as = a[p], a[q], a[r], a[s]
        t3 = ap * M3[q, j] + aq * M3[p, j] + ar * M3t[i, s] + as * M3t[i, r]
        t2 = ap * aq * M2[s, r] +
             ar * as * M2[q, p] +
             ap * ar * M2[q, s] +
             aq * as * M2[p, r] +
             ap * as * M2[q, r] +
             aq * ar * M2[s, p]
        out[i, j] += M4[i, j] - t3 + t2 + m * ap * aq * ar * as
    end
    return out
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an estimator whose configuration no incremental fit reproduces, naming the field that refuses it.

A partial fit keeps the running sample mean and the central co-moments about it, and nothing else of the sample. Two configuration choices break that. The function refuses both, and it does not approximate either.

  - **Observation weights.** A weights vector describes a sample of a fixed length, and a [`DynamicAbstractWeights`](@ref) derives every weight from the length of the sample. Either way a new observation reweights every past one, so no state written before it is still valid.
  - **A mean estimator that is not the plain sample mean.** The state carries the running sample mean. An estimator whose centre is a shrunk mean, a median or a custom value needs a centre that the state does not hold.

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
              ArgumentError("$name cannot be fitted incrementally with `me = $(typeof(me))`, because a partial-fit state carries the running sample mean and no other centre. The available-case fit of a `CoveragePolicy` centres each asset on that mean too. Set `me = SimpleExpectedReturns()`, or clear `cvg` and run the batch verb."))
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

The count, the mean and the second accumulator take the merge of [`chan_merge`](@ref). [`shift_comoment3!`](@ref) moves the third accumulator of each block to the common mean, and it adds the two shifted accumulators into one array. For one series, the result is Proposition 2.1 of [pebay2008](@cite) at ``p = 3``.

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
  - [`shift_comoment3!`](@ref)
  - [`chan_merge`](@ref)

# References

  - $(ref_dict[:pebay2008]) Proposition 2.1, page 10. $(ref_dict[:no_original_source])
"""
function merge_states(a::CoskewnessPartialFitState, b::CoskewnessPartialFitState)
    assert_mergeable_states(a, b)
    n, mu, M2 = chan_merge(a.n, a.mu, a.M2, b.n, b.mu, b.M2)
    M3 = zeros(promote_type(eltype(mu), eltype(a.M3), eltype(b.M3)), size(a.M3))
    shift_comoment3!(M3, a.n, mu .- a.mu, a.M2, a.M3)
    shift_comoment3!(M3, b.n, mu .- b.mu, b.M2, b.M3)
    return CoskewnessPartialFitState(n, mu, M2, M3)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`CoskewnessPartialFitState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface. The count is a scalar and passes through, and the method copies the running mean and the two accumulators. This family overrides [`partial_fit`](@ref), so the value form of that interface never calls this method on it. A caller who copies a state by hand still gets a copy.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::CoskewnessPartialFitState`: A fresh state, equal to `x`, whose `mu`, `M2` and `M3` are fresh arrays.

# Related

  - [`CoskewnessPartialFitState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::CoskewnessPartialFitState)
    return CoskewnessPartialFitState(x.n, copy(x.mu), copy(x.M2), copy(x.M3))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`CoskewnessPartialFitState`](@ref) to the selected assets.

A third co-moment reads three assets' observations and no fourth, so the sub-tensor over a subset of assets is the state of that subset, entry for entry, and the count passes through. `M3` is `assets × assets²`, and its column for the pair ``(p, q)`` is ``(p - 1) N + q``, so the slice keeps the rows `i` and the columns of every pair drawn from `i`. [`fourth_moment_index_generator`](@ref) builds that column set, in the order the sliced universe numbers its own pairs.

The slice copies by index and does not `view`, so a later [`partial_fit!`](@ref) on the viewed estimator writes into arrays of its own. The copy is small. A cluster's sub-tensor is cubic in the cluster size, and the full one is cubic in the universe size.

# Arguments

  - `x`: The state to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `state::CoskewnessPartialFitState`: The state of the same sample over the selected assets.

# Related

  - [`CoskewnessPartialFitState`](@ref)
  - [`port_opt_view`](@ref)
  - [`fourth_moment_index_generator`](@ref)
  - [`partial_fit!`](@ref)
"""
function port_opt_view(x::CoskewnessPartialFitState, i, args...)
    idx = fourth_moment_index_generator(length(x.mu), i)
    return CoskewnessPartialFitState(x.n, x.mu[i], x.M2[i, i], x.M3[i, idx])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds two [`CokurtosisPartialFitState`](@ref) fitted on disjoint blocks into the state of the concatenated block.

The companion of the [`CoskewnessPartialFitState`](@ref) merge, one order higher. The merge shifts the fourth accumulator before the third, because [`shift_comoment4!`](@ref) reads the third accumulator of the block about its own mean. For one series, the result is Proposition 2.1 of [pebay2008](@cite) at ``p = 4``.

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
  - [`shift_comoment4!`](@ref)
  - [`chan_merge`](@ref)

# References

  - $(ref_dict[:pebay2008]) Proposition 2.1, page 10. $(ref_dict[:no_original_source])
"""
function merge_states(a::CokurtosisPartialFitState, b::CokurtosisPartialFitState)
    assert_mergeable_states(a, b)
    n, mu, M2 = chan_merge(a.n, a.mu, a.M2, b.n, b.mu, b.M2)
    da = mu .- a.mu
    db = mu .- b.mu
    M4 = zeros(promote_type(eltype(mu), eltype(a.M4), eltype(b.M4)), size(a.M4))
    shift_comoment4!(M4, a.n, da, a.M2, a.M3, a.M4)
    shift_comoment4!(M4, b.n, db, b.M2, b.M3, b.M4)
    M3 = zeros(promote_type(eltype(mu), eltype(a.M3), eltype(b.M3)), size(a.M3))
    shift_comoment3!(M3, a.n, da, a.M2, a.M3)
    shift_comoment3!(M3, b.n, db, b.M2, b.M3)
    return CokurtosisPartialFitState(n, mu, M2, M3, M4)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`CokurtosisPartialFitState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface. The count is a scalar and passes through, and the method copies the running mean and the three accumulators. `M4` is `assets² × assets²`, so this method is the expensive one of that interface. That is why the family overrides [`partial_fit`](@ref), which would pay this copy on every fold.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::CokurtosisPartialFitState`: A fresh state, equal to `x`, whose `mu`, `M2`, `M3` and `M4` are fresh arrays.

# Related

  - [`CokurtosisPartialFitState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::CokurtosisPartialFitState)
    return CokurtosisPartialFitState(x.n, copy(x.mu), copy(x.M2), copy(x.M3), copy(x.M4))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Slices a [`CokurtosisPartialFitState`](@ref) to the selected assets.

A fourth co-moment reads four assets' observations and no fifth, so the sub-tensor over a subset of assets is the state of that subset, entry for entry, and the count passes through. `M4` is `assets² × assets²`, and both of its axes are numbered by the pair ``(p, q)`` at ``(p - 1) N + q``, so the slice keeps the columns of every pair drawn from `i` on both axes. `M3` keeps the rows `i` and the same column set.

The slice copies by index and does not `view`, so a later [`partial_fit!`](@ref) on the viewed estimator writes into arrays of its own. The copy is small next to the state it comes from. A cluster's sub-tensor is quartic in the cluster size, so ten clusters of ten assets copy 80 KB each out of the 800 MB that a hundred-asset state holds.

# Arguments

  - `x`: The state to slice.
  - `i`: Index or indices of the assets to keep.
  - `args...`: Additional positional arguments (ignored).

# Returns

  - `state::CokurtosisPartialFitState`: The state of the same sample over the selected assets.

# Related

  - [`CokurtosisPartialFitState`](@ref)
  - [`port_opt_view`](@ref)
  - [`fourth_moment_index_generator`](@ref)
  - [`partial_fit!`](@ref)
"""
function port_opt_view(x::CokurtosisPartialFitState, i, args...)
    idx = fourth_moment_index_generator(length(x.mu), i)
    return CokurtosisPartialFitState(x.n, x.mu[i], x.M2[i, i], x.M3[i, idx], x.M4[idx, idx])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds a block of observations into the partial-fit state of a [`Coskewness`](@ref) estimator under [`FullMoment`](@ref).

The estimator that carries no state adopts the block's state. The estimator that carries one merges the two, so the state after any sequence of blocks is the state of the concatenated sample. One row per call is exact. The block of one observation carries zero accumulators, and the whole correction is the shift of the running state to the new mean.

The method builds a new state and does not write into the arrays of the old one, because [`merge_states`](@ref) holds the one merge of two blocks. A caller must not rely on that. The verb promises nothing about an estimator kept from before the call, and [`partial_fit`](@ref) is the verb that does. This family overrides [`partial_fit`](@ref), because a copy of a state that the fold never writes into buys nothing.

The state keeps no observation, so `coskewness(ske)` runs `ske.mp` from the shape of the sample, with no sample. The steps `:pdm`, `:dn` and `:dt` of a [`MatrixProcessing`](@ref) read no value of the sample. An `:alg` step can read the values, as [`LoGo`](@ref) does under a variation of information distance, and [`reads_sample_values`](@ref) states whether it does. So the method refuses an `mp` that does not run from the shape of the sample, before it folds a row. [`Online`](@ref) keeps the observations and runs such a step. The default `LoGo()` reads no value of the sample, so it folds.

# Algorithm

 1. Refuse the configuration no incremental fit reproduces, with [`assert_partial_fittable`](@ref) and [`assert_shape_only_matrix_processing`](@ref).
 2. Orient `X` to observations × assets with [`dims_oriented`](@ref), which validates `dims`, and write `NaN` in each cell that `active_mask` turns off with [`inactive_as_nonfinite`](@ref). Return the estimator unchanged when the block holds no observation, because the fold of an empty sample is the identity, and the mean of an empty block is `NaN`, which the merge would carry into every accumulator.
 3. Build the state of the block alone, with [`comoment_block`](@ref).
 4. Merge it into the state the estimator carries, unless the estimator carries none.
 5. Rebuild the estimator with the merged state in `cache` through [`rebuild_struct`](@ref), and return it.

# Arguments

  - `ske`: Coskewness estimator with a [`FullMoment`](@ref) moment algorithm.
  - `X`: Block of observations (observations × assets).
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the Asset Panel, of the shape of `X`, or `nothing`. A plain state carries one count for every asset, so it cannot skip an inactive cell. [`inactive_as_nonfinite`](@ref) writes `NaN` there, and the asset then answers `NaN`, as the Asset Panel method answers an asset outside the Coverage Universe of the same rows.

# Validation

  - `ske.w` is `nothing` and `ske.me` is an unweighted [`SimpleExpectedReturns`](@ref).
  - `ske.mp` runs from the shape of the sample, as [`assert_shape_only_matrix_processing`](@ref) states. An `ArgumentError` is thrown otherwise.
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
function partial_fit!(ske::Coskewness{<:Any, <:Any, <:FullMoment, <:Any, Nothing,
                                      <:Option{<:CoskewnessPartialFitState}}, X::MatNum;
                      dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing)
    assert_partial_fittable(ske.me, ske.w, "Coskewness")
    assert_shape_only_matrix_processing(ske.mp)
    X = inactive_as_nonfinite(dims_oriented(dims, X), dims_oriented(dims, active_mask))
    if iszero(size(X, 1))
        return ske
    end
    n, mu, M2, M3, _ = comoment_block(X)
    state = CoskewnessPartialFitState(n, mu, M2, M3)
    if !isnothing(ske.cache)
        state = merge_states(ske.cache, state)
    end
    return rebuild_struct(ske, (; cache = state))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds one observation into the partial-fit state of a [`Coskewness`](@ref) estimator under [`FullMoment`](@ref).

The single-observation arm of the [`partial_fit!`](@ref) interface. The entries of `x` are the assets, so the observation is the block of one row.

# Algorithm

 1. Write `NaN` in each cell of `x` that `active_mask` turns off, with [`inactive_as_nonfinite`](@ref).
 2. Reshape `x` to a block of one row, and forward it to the block method of [`partial_fit!`](@ref).

# Arguments

  - `ske`: Coskewness estimator with a [`FullMoment`](@ref) moment algorithm.
  - `x`: One observation, whose entries are the assets.
  - `active_mask`: The active mask of the Asset Panel at this observation, one entry per asset, or `nothing`. An inactive cell reads as a non-finite return, as in the block method.

# Returns

  - `ske::Coskewness`: The estimator carrying the folded state.

# Related

  - [`Coskewness`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(ske::Coskewness{<:Any, <:Any, <:FullMoment, <:Any, Nothing,
                                      <:Option{<:CoskewnessPartialFitState}}, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing)
    return partial_fit!(ske, reshape(inactive_as_nonfinite(x, active_mask), 1, length(x)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`Coskewness`](@ref) method of [`partial_fit`](@ref). Forwards to [`partial_fit!`](@ref), because the fold already writes into no array of the state it was given.

This family overrides the generic method. The generic method copies the state before it folds, and the copy gives the verb its value semantics. The fold of this family reaches [`merge_states`](@ref), which reads the two states and builds a third. So the fold leaves the given estimator as it was, and the copy buys nothing. The override stays for a measured loss. At 30 assets, one row takes 0.11 ms and 0.46 MiB through it, and 0.13 ms and 0.67 MiB through the generic method.

# Algorithm

 1. Forward `X` and every keyword to [`partial_fit!`](@ref), and return the estimator it gives.

# Arguments

  - `ske`: Coskewness estimator with a [`FullMoment`](@ref) moment algorithm.
  - `args...`: The observations, forwarded to [`partial_fit!`](@ref).
  - `kwargs...`: Additional keyword arguments, forwarded to [`partial_fit!`](@ref).

# Returns

  - `ske::Coskewness`: A new estimator carrying the folded state.

# Related

  - [`Coskewness`](@ref)
  - [`partial_fit`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit(ske::Coskewness{<:Any, <:Any, <:FullMoment}, args...; kwargs...)
    return partial_fit!(ske, args...; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an incremental fit of a [`Coskewness`](@ref) estimator under [`SemiMoment`](@ref).

[`SemiMoment`](@ref) clips every positive deviation to zero before the moment, and the centre it clips against is a statistic of the whole sample. A new observation moves that centre, so every past clip moves with it and a past observation's membership of the clipped set flips. No state written before that observation is still valid.

# Arguments

  - `ske`: Coskewness estimator with a [`SemiMoment`](@ref) moment algorithm.
  - `::VecNum_MatNum`: The observations, ignored.
  - `kwargs...`: Ignored.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`Coskewness`](@ref)
  - [`SemiMoment`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(::Coskewness{<:Any, <:Any, <:SemiMoment, <:Any, <:Any,
                                   <:Option{<:CoskewnessPartialFitState}}, ::VecNum_MatNum;
                      kwargs...)
    return throw(ArgumentError("a `Coskewness` estimator with a `SemiMoment` moment algorithm cannot be fitted incrementally, because the clip against the sample mean moves when the mean moves, so a past observation's membership of the clipped set flips. Use a `FullMoment` algorithm, or run the batch verb."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds a block of observations into the partial-fit state of a [`Cokurtosis`](@ref) estimator under [`FullMoment`](@ref).

The companion of the [`Coskewness`](@ref) method, one order higher. The largest field of the state that it writes is `assets² × assets²`, and [`CokurtosisPartialFitState`](@ref) states the memory that this costs. The method builds a new state and does not write into the arrays of the old one, for the reason that the [`Coskewness`](@ref) method states. It overrides [`partial_fit`](@ref) for the same reason, and it refuses an `mp` that does not run from the shape of the sample for the same reason too.

# Algorithm

 1. Refuse the configuration no incremental fit reproduces, with [`assert_partial_fittable`](@ref) and [`assert_shape_only_matrix_processing`](@ref).
 2. Orient `X` to observations × assets with [`dims_oriented`](@ref), which validates `dims`, and write `NaN` in each cell that `active_mask` turns off with [`inactive_as_nonfinite`](@ref). Return the estimator unchanged when the block holds no observation, for the reason that the [`Coskewness`](@ref) method states.
 3. Build the state of the block alone, with [`comoment_block`](@ref), and square its pairwise deviation product for the fourth accumulator.
 4. Merge it into the state the estimator carries, unless the estimator carries none.
 5. Rebuild the estimator with the merged state in `cache` through [`rebuild_struct`](@ref), and return it.

# Arguments

  - `kte`: Cokurtosis estimator with a [`FullMoment`](@ref) moment algorithm.
  - `X`: Block of observations (observations × assets).
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the Asset Panel, of the shape of `X`, or `nothing`. A plain state carries one count for every asset, so it cannot skip an inactive cell. [`inactive_as_nonfinite`](@ref) writes `NaN` there, and the asset then answers `NaN`, as the Asset Panel method answers an asset outside the Coverage Universe of the same rows.

# Validation

  - `kte.w` is `nothing` and `kte.me` is an unweighted [`SimpleExpectedReturns`](@ref).
  - `kte.mp` runs from the shape of the sample, as [`assert_shape_only_matrix_processing`](@ref) states. An `ArgumentError` is thrown otherwise.
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
function partial_fit!(kte::Cokurtosis{<:Any, <:Any, <:FullMoment, <:Any, Nothing,
                                      <:Option{<:CokurtosisPartialFitState}}, X::MatNum;
                      dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing)
    assert_partial_fittable(kte.me, kte.w, "Cokurtosis")
    assert_shape_only_matrix_processing(kte.mp)
    X = inactive_as_nonfinite(dims_oriented(dims, X), dims_oriented(dims, active_mask))
    if iszero(size(X, 1))
        return kte
    end
    n, mu, M2, M3, z = comoment_block(X)
    state = CokurtosisPartialFitState(n, mu, M2, M3, transpose(z) * z)
    if !isnothing(kte.cache)
        state = merge_states(kte.cache, state)
    end
    return rebuild_struct(kte, (; cache = state))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds one observation into the partial-fit state of a [`Cokurtosis`](@ref) estimator under [`FullMoment`](@ref).

The single-observation arm of the [`partial_fit!`](@ref) interface. The entries of `x` are the assets, so the observation is the block of one row.

# Algorithm

 1. Write `NaN` in each cell of `x` that `active_mask` turns off, with [`inactive_as_nonfinite`](@ref).
 2. Reshape `x` to a block of one row, and forward it to the block method of [`partial_fit!`](@ref).

# Arguments

  - `kte`: Cokurtosis estimator with a [`FullMoment`](@ref) moment algorithm.
  - `x`: One observation, whose entries are the assets.
  - `active_mask`: The active mask of the Asset Panel at this observation, one entry per asset, or `nothing`. An inactive cell reads as a non-finite return, as in the block method.

# Returns

  - `kte::Cokurtosis`: The estimator carrying the folded state.

# Related

  - [`Cokurtosis`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(kte::Cokurtosis{<:Any, <:Any, <:FullMoment, <:Any, Nothing,
                                      <:Option{<:CokurtosisPartialFitState}}, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing)
    return partial_fit!(kte, reshape(inactive_as_nonfinite(x, active_mask), 1, length(x)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

[`Cokurtosis`](@ref) method of [`partial_fit`](@ref). Forwards to [`partial_fit!`](@ref), because the fold already writes into no array of the state it was given.

The companion of the [`Coskewness`](@ref) method, and the one that pays for the override. The generic method copies `M4`, which is `assets² × assets²`, before every fold. That is 800 MB per call at a hundred assets, for a copy that no caller reads. The override stays for that measured loss. At 30 assets, one row allocates 13.2 MiB through it, and 19.6 MiB through the generic method.

# Algorithm

 1. Forward `X` and every keyword to [`partial_fit!`](@ref), and return the estimator it gives.

# Arguments

  - `kte`: Cokurtosis estimator with a [`FullMoment`](@ref) moment algorithm.
  - `args...`: The observations, forwarded to [`partial_fit!`](@ref).
  - `kwargs...`: Additional keyword arguments, forwarded to [`partial_fit!`](@ref).

# Returns

  - `kte::Cokurtosis`: A new estimator carrying the folded state.

# Related

  - [`Cokurtosis`](@ref)
  - [`partial_fit`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit(kte::Cokurtosis{<:Any, <:Any, <:FullMoment}, args...; kwargs...)
    return partial_fit!(kte, args...; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses an incremental fit of a [`Cokurtosis`](@ref) estimator under [`SemiMoment`](@ref).

The same refusal as the [`Coskewness`](@ref) one, for the same reason: [`SemiMoment`](@ref) clips against a centre that a new observation moves, so a past observation's membership of the clipped set flips.

# Arguments

  - `kte`: Cokurtosis estimator with a [`SemiMoment`](@ref) moment algorithm.
  - `::VecNum_MatNum`: The observations, ignored.
  - `kwargs...`: Ignored.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`Cokurtosis`](@ref)
  - [`SemiMoment`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit!(::Cokurtosis{<:Any, <:Any, <:SemiMoment, <:Any, <:Any,
                                   <:Option{<:CokurtosisPartialFitState}}, ::VecNum_MatNum;
                      kwargs...)
    return throw(ArgumentError("a `Cokurtosis` estimator with a `SemiMoment` moment algorithm cannot be fitted incrementally, because the clip against the sample mean moves when the mean moves, so a past observation's membership of the clipped set flips. Use a `FullMoment` algorithm, or run the batch verb."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the coskewness tensor and its negative spectral skewness matrix out of a partial-fit state.

The method of [`coskewness`](@ref) that makes the estimate from a Partial Fit State. It returns the pair the batch method returns, to machine precision, for the sample the state was fitted on.

Only the shape of that sample survives a partial fit, so the matrix processing estimator runs from the shape, through the shape method of [`matrix_processing!`](@ref). Of the steps that [`partial_fit!`](@ref) lets through, only `:dn` reads the sample, and `:dn` reads only the ratio `T / N`, which the state keeps. The `:alg` step gets no sample, as in the estimate of a [`PortfolioOptimisersCovariance`](@ref). An `:alg` step that [`partial_fit!`](@ref) lets through reads no value of the sample, as [`reads_sample_values`](@ref) states. So the two routes agree. [`partial_fit!`](@ref) refuses an `:alg` step that can read the values of the sample, and a wrong `false` answer of [`reads_sample_values`](@ref) raises an error here.

# Algorithm

 1. Refuse a configuration the state no longer matches, with [`assert_partial_fittable`](@ref). [`factory`](@ref) carries the state and replaces `w`, so an estimator that says weighted may hold a state fitted unweighted. The state stays on the estimator, so a caller who restores `w = nothing` reads it again.
 2. Divide the third accumulator by the observation count, giving the coskewness tensor.
 3. Reduce it with [`negative_spectral_coskewness`](@ref), under `ske.mp`, with no sample and the observation count of the state.

# Arguments

  - `ske`: Coskewness estimator with a [`FullMoment`](@ref) moment algorithm.
  - `state`: Partial-fit state written by [`partial_fit!`](@ref).

# Validation

  - `ske.w` is `nothing` and `ske.me` is an unweighted [`SimpleExpectedReturns`](@ref). An `ArgumentError` is thrown otherwise.

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
    assert_partial_fittable(ske.me, ske.w, "Coskewness")
    cskew = state.M3 ./ state.n
    return cskew, negative_spectral_coskewness(cskew, nothing, ske.mp, state.n)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the coskewness tensor and its negative spectral skewness matrix out of the state the estimator carries.

The one-argument forwarder of [`coskewness`](@ref). It reads `ske.cache`, which [`partial_fit!`](@ref) writes. It refuses an estimator that has folded no observation.

# Algorithm

 1. Refuse the estimator whose `cache` is `nothing`.
 2. Forward to the state method of [`coskewness`](@ref).

# Arguments

  - `ske`: Coskewness estimator.

# Validation

  - `ske.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:cskew])
  - $(ret_dict[:cskewV])

# Related

  - [`Coskewness`](@ref)
  - [`partial_fit!`](@ref)
"""
function coskewness(ske::Coskewness)
    @argcheck(!isnothing(ske.cache),
              ArgumentError("this `Coskewness` estimator carries no partial-fit state, so there is nothing to read out. Call `partial_fit!` first, or pass a data matrix."))
    return coskewness(ske, ske.cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the square cokurtosis matrix out of a partial-fit state.

The method of [`cokurtosis`](@ref) that makes the estimate from a Partial Fit State. It returns the matrix the batch method returns, to machine precision, for the sample the state was fitted on.

Only the shape of that sample survives a partial fit, so the matrix processing estimator runs from the shape, through the shape method of [`matrix_processing_block!`](@ref). Of the steps that [`partial_fit!`](@ref) lets through, only `:dn` reads the sample, and `:dn` reads only the ratio `T / N`, which the state keeps. `N` is the number of assets, and not the `N^2` asset pairs of the side of the matrix, as in the batch verb. On a frame, `N` is the number of assets of the block, as [`block_assets`](@ref) states, so the ratio is the ratio of the batch verb over the columns that the block covers. The `:alg` step gets no sample, as in the estimate of a [`PortfolioOptimisersCovariance`](@ref). An `:alg` step that [`partial_fit!`](@ref) lets through reads no value of the sample, as [`reads_sample_values`](@ref) states. So the two routes agree. [`partial_fit!`](@ref) refuses an `:alg` step that can read the values of the sample, and a wrong `false` answer of [`reads_sample_values`](@ref) raises an error here.

# Algorithm

 1. Refuse a configuration the state no longer matches, with [`assert_partial_fittable`](@ref). [`factory`](@ref) carries the state and replaces `w`, so an estimator that says weighted may hold a state fitted unweighted. The state stays on the estimator, so a caller who restores `w = nothing` reads it again.
 2. Divide the fourth accumulator by the observation count, giving the cokurtosis matrix.
 3. Process it in place through the shape method of [`matrix_processing_block!`](@ref), under `kte.mp`, the observation count of the state and the number of assets. The method takes the block arm for the reason that `cov(ce)` with no data takes it. A plain fold over a changing universe answers `NaN` at every pair that names an asset outside the Coverage Universe. A positive definite repair over that frame then meets a refusal from LAPACK, not a named error. The block arm repairs the finite block and leaves the frame for [`investable_mask`](@ref) to read, as the reduce-and-expand of the batch verb does. A complete matrix runs the plain arm over the whole of it.

# Arguments

  - `kte`: Cokurtosis estimator with a [`FullMoment`](@ref) moment algorithm.
  - `state`: Partial-fit state written by [`partial_fit!`](@ref).

# Validation

  - `kte.w` is `nothing` and `kte.me` is an unweighted [`SimpleExpectedReturns`](@ref). An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:ckurt])

# Related

  - [`Cokurtosis`](@ref)
  - [`CokurtosisPartialFitState`](@ref)
  - [`partial_fit!`](@ref)
  - [`matrix_processing_block!`](@ref)
"""
function cokurtosis(kte::Cokurtosis{<:Any, <:Any, <:FullMoment},
                    state::CokurtosisPartialFitState)
    assert_partial_fittable(kte.me, kte.w, "Cokurtosis")
    ckurt = state.M4 ./ state.n
    matrix_processing_block!(kte.mp, ckurt, state.n, length(state.mu))
    return ckurt
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the square cokurtosis matrix out of the state the estimator carries.

The one-argument forwarder of [`cokurtosis`](@ref). It reads `kte.cache`, which [`partial_fit!`](@ref) writes. It refuses an estimator that has folded no observation.

# Algorithm

 1. Refuse the estimator whose `cache` is `nothing`.
 2. Forward to the state method of [`cokurtosis`](@ref).

# Arguments

  - `kte`: Cokurtosis estimator.

# Validation

  - `kte.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:ckurt])

# Related

  - [`Cokurtosis`](@ref)
  - [`partial_fit!`](@ref)
"""
function cokurtosis(kte::Cokurtosis)
    @argcheck(!isnothing(kte.cache),
              ArgumentError("this `Cokurtosis` estimator carries no partial-fit state, so there is nothing to read out. Call `partial_fit!` first, or pass a data matrix."))
    return cokurtosis(kte, kte.cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads a coskewness tensor out of a [`SampleBufferState`](@ref), by running the batch verb over the observations the buffer holds.

The method of `coskewness` that makes the estimate from a Sample Buffer. An estimator wrapped in [`Online`](@ref) carries a buffer and not a family state, so it takes no exact fold. Its estimate is the batch fit over the rows that the buffer holds. An uncapped buffer holds every observation folded so far, and a capped one holds the last `max_history` of them. The method runs no [`assert_partial_fittable`](@ref) check of its own. The batch verb answers configurations that an incremental fold cannot, and a wrapper is the way a caller reaches them. The buffer also holds the mask of each observation that it folded. So the batch verb gets the mask that explains the rows, and a wrapped estimator under a [`CoveragePolicy`](@ref) answers what an unwrapped one answers.

The call with no data, `coskewness(ske)`, fills `state` with the state that the estimator carries, which [`partial_fit_cache`](@ref) reads out of the `cache` field. The call then goes to the method of `coskewness` for the type of that state. This is the call with no data of an estimator that has no such call of its own. A family that folds exactly has a call with no data of its own, so the default serves the estimators that fold by buffering alone, such as [`WindowedCoskewness`](@ref).

# Arguments

  - $(arg_dict[:ske])
  - `state`: The buffer to read. The default is the state that `ske` carries.

# Validation

  - With no `state`, the estimator carries a state. An `ArgumentError` is thrown otherwise.

# Returns

  - `(cskew, V)::Tuple`: The coskewness tensor of the observations the buffer holds, and its negative spectral shape.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer`](@ref)
  - [`partial_fit_cache`](@ref)
  - [`partial_fit!`](@ref)
"""
function coskewness(ske::CoskewnessEstimator,
                    state::SampleBufferState = partial_fit_cache(ske))
    return coskewness(ske, sample_buffer(state); sample_buffer_kwargs(state)...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads a cokurtosis matrix out of a [`SampleBufferState`](@ref), by running the batch verb over the observations the buffer holds.

The method of `cokurtosis` that makes the estimate from a Sample Buffer. An estimator wrapped in [`Online`](@ref) carries a buffer and not a family state, so it takes no exact fold. Its estimate is the batch fit over the rows that the buffer holds. An uncapped buffer holds every observation folded so far, and a capped one holds the last `max_history` of them. The method runs no [`assert_partial_fittable`](@ref) check of its own. The batch verb answers configurations that an incremental fold cannot, and a wrapper is the way a caller reaches them. The buffer also holds the mask of each observation that it folded. So the batch verb gets the mask that explains the rows, and a wrapped estimator under a [`CoveragePolicy`](@ref) answers what an unwrapped one answers.

The call with no data, `cokurtosis(kte)`, fills `state` with the state that the estimator carries, which [`partial_fit_cache`](@ref) reads out of the `cache` field. The call then goes to the method of `cokurtosis` for the type of that state, as the call with no data of [`coskewness`](@ref) does. The default serves the estimators that fold by buffering alone, such as [`WindowedCokurtosis`](@ref).

# Arguments

  - $(arg_dict[:kte])
  - `state`: The buffer to read. The default is the state that `kte` carries.

# Validation

  - With no `state`, the estimator carries a state. An `ArgumentError` is thrown otherwise.

# Returns

  - `ckurt::MatNum`: Cokurtosis matrix of the observations the buffer holds.

# Related

  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`sample_buffer`](@ref)
  - [`partial_fit_cache`](@ref)
  - [`partial_fit!`](@ref)
"""
function cokurtosis(kte::CokurtosisEstimator,
                    state::SampleBufferState = partial_fit_cache(kte))
    return cokurtosis(kte, sample_buffer(state); sample_buffer_kwargs(state)...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reads the expected returns out of the partial-fit state that an expected returns estimator carries.

This is the call with no data of an estimator that has no such call of its own. It reads the state out of the `cache` field with [`partial_fit_cache`](@ref), and it gives the state to the method of `Statistics.mean` that takes one. A [`SampleBufferState`](@ref), which [`Online`](@ref) seeds, runs the batch verb over the rows of the buffer. A family that folds exactly has a call with no data of its own, so this method serves the estimators that fold by buffering alone, such as [`WindowedExpectedReturns`](@ref).

# Arguments

  - $(arg_dict[:me])

# Validation

  - `me` carries a state. An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:mu])

# Related

  - [`partial_fit_cache`](@ref)
  - [`SampleBufferState`](@ref)
  - [`Online`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.mean(me::AbstractExpectedReturnsEstimator)
    return Statistics.mean(me, partial_fit_cache(me))
end

# A co-moment folds under `FullMoment` with no Coverage Policy, when its `mp` runs from the
# shape of the sample. The `SemiMoment` arm refuses for the reason that `Covariance` does, and
# `partial_fit!` refuses an `mp.alg` that reads the values of the sample. Both take the
# buffer route of the default `supports_partial_fit`, as `PortfolioOptimisersCovariance` does.
function supports_partial_fit(ske::Coskewness{<:Any, <:Any, <:FullMoment, <:Any, Nothing})
    return isa(ske.cache, SampleBufferState) || supports_shape_processing(ske.mp)
end
function supports_partial_fit(kte::Cokurtosis{<:Any, <:Any, <:FullMoment, <:Any, Nothing})
    return isa(kte.cache, SampleBufferState) || supports_shape_processing(kte.mp)
end
