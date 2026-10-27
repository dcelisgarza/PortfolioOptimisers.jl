"""
    partial_fit!(est, X)

Folds observations into an estimator's partial-fit state, and returns the estimator.

An incremental fit reads each observation once. It keeps the data that the estimate needs in an [`AbstractPartialFitState`](@ref), so a later call starts from that state and does not read the sample again. The state is in the `cache` field of the estimator, which holds `nothing` until the first call. That field is the one Result that an estimator holds.

The states of the sample mean, variance, covariance, coskewness and cokurtosis also keep the accuracy of the estimate. Each one adds an observation by Welford's recursion and a block by [`chan_merge`](@ref), and both centre each increment on the running mean. The textbook one-pass formula subtracts two large sums instead. Table 1 of [chan1983](@cite) bounds the relative error of the variance by ``N \\kappa^2 u`` for the textbook formula, and by ``N \\kappa u`` for the updating formula. ``N`` is the number of observations, ``u`` is the unit roundoff, and ``\\kappa`` is about the mean divided by the standard deviation. So a price series whose mean is 1000 times its standard deviation gives the textbook formula a bound 1000 times larger.

Each family writes a method of this verb. The method writes into the array fields of the state where it can. It builds a new state with the new scalar fields through the positional constructor of the state. So it returns a **new** estimator, and the caller must rebind it.

The verb promises nothing about an estimator that the caller kept from before the call. The `!` in the name shows this, and three faults follow from it. [`partial_fit`](@ref) has none of them, because it folds a copy of each state.

  - A kept estimator holds a state whose arrays changed and whose count did not, so it describes neither the old sample nor the new one.
  - Two folds that start from one estimator write into the same arrays, so each fold changes the result of the other.
  - A family whose fold builds a new state leaves the kept estimator correct. No caller can rely on that.

A batch verb ignores the state. `var(ce, X)` fits `X` alone, so an estimator that holds a state still gives the estimate of any `X`.

# Interfaces

A family that answers this verb implements two methods:

  - `partial_fit!(est, X::MatNum; dims::Int = 1, kwargs...) -> est`: Folds every observation of `X`, in order.
  - `partial_fit!(est, x::VecNum; kwargs...) -> est`: Folds one observation, whose entries are the assets.

A family that folds also states it through [`supports_partial_fit`](@ref):

  - `supports_partial_fit(est::MyEstimator) -> Bool`: Returns `true`.

The fallback returns `true` only for an estimator whose `cache` holds a [`SampleBufferState`](@ref). The carry fold of a [`CrossSectionalFactorPrior`](@ref) reads this answer. At each step, it fits a member that returns `false` again, over every row. Under [`FoldOnly`](@ref), the constructor of the prior refuses such a `ve`.

# Arguments

  - `est`: The estimator whose state the verb folds.
  - `X`: Observations to fold. A matrix holds one observation per row when `dims == 1`, and one per column when `dims == 2`. A vector is a single observation across the assets.

# Returns

  - `est`: The estimator, with its `cache` field rebound to the state after the last observation.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`partial_fit`](@ref)
  - [`merge_states`](@ref)
  - [`chan_merge`](@ref)

# References

  - $(ref_dict[:chan1983]) Table 1. $(ref_dict[:no_original_source])
"""
function partial_fit! end
"""
    partial_fit(est, args...; kwargs...)

Folds observations into a copy of the partial-fit state of an estimator, and returns the estimator that holds the copy.

This is the value form of [`partial_fit!`](@ref), as [`matrix_processing`](@ref) is the value form of [`matrix_processing!`](@ref). The verb does not change the estimator that it gets, or the state of that estimator. So two folds that start from one estimator do not change each other.

One generic method applies to every family of the `partial_fit!` interface, so a family writes no method for it. A family whose fold always builds a new state can add a method, to skip a copy that nothing reads. [`Coskewness`](@ref) and [`Cokurtosis`](@ref) add one.

# Interfaces

A family that overrides this verb implements the two methods the family already answers under [`partial_fit!`](@ref):

  - `partial_fit(est, X::MatNum; dims::Int = 1, kwargs...) -> est`: Folds every observation of `X`, in order.
  - `partial_fit(est, x::VecNum; kwargs...) -> est`: Folds one observation, whose entries are the assets.

# Arguments

  - `est`: The estimator whose state the verb folds.
  - `X`: Observations to fold. A matrix holds one observation per row when `dims == 1`, and one per column when `dims == 2`. A vector is a single observation across the assets.

# Returns

  - `est`: A new estimator, whose states hold the fold after the last observation.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`partial_fit!`](@ref)
  - [`copy_states`](@ref)
  - [`merge_states`](@ref)
"""
function partial_fit end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Generic method of [`partial_fit`](@ref). Copies every state that the estimator tree holds, and folds the observations into the copies.

Every family of the `partial_fit!` interface uses this method, because the copy and the fold are the same two steps for each state. The copy is [`copy_states`](@ref), which [`Resume`](@ref) also calls at entry. It copies the `cache` of the estimator when that field holds a state. It goes into every field that holds an estimator, and it rebuilds each outer estimator whose fields changed.

So the method also applies to an outer estimator that folds through the states of its members. A [`HighOrderPriorEstimator`](@ref) folds its `pe`, `ske` and `kte`, and a hierarchical optimiser folds its `opt.pe`. The states of the kept outer estimator do not change, as for a leaf. The cost is one copy of each state in the tree. For a second-order family, that copy costs as much as the fold of one observation.

An estimator with no incremental fit holds no state, so the walk returns it with no change. [`partial_fit!`](@ref) then refuses it, and the message names the type of the estimator.

# Algorithm

 1. Copy every state of the estimator tree with [`copy_states`](@ref), giving a rebuilt tree. A tree that holds no state comes back with no change, because the fold makes a new state.
 2. Fold the observations into the copies with [`partial_fit!`](@ref), and return the estimator that it gives.

# Arguments

  - `est`: The estimator whose state the verb folds.
  - `args...`: The observations, forwarded to [`partial_fit!`](@ref).
  - `kwargs...`: Additional keyword arguments, forwarded to [`partial_fit!`](@ref).

# Returns

  - `est`: A new estimator, whose states hold the fold after the last observation.

# Related

  - [`partial_fit`](@ref)
  - [`partial_fit!`](@ref)
  - [`copy_states`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function partial_fit(est::Union{AbstractEstimator, StatsBase.CovarianceEstimator}, args...;
                     kwargs...)
    return partial_fit!(copy_states(est), args...; kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Removes the partial-fit state from an estimator whose observations a view selects.

[`obs_weights_view`](@ref) selects observations, and a state describes the observations of its fit. A state has no slice on that axis, because no numerically stable operation removes an observation from a running accumulator. For the same reason, [`partial_fit!`](@ref) refuses an estimator under a [`RollingWindow`](@ref). So the method returns `nothing` in place of the state. The viewed estimator then refuses a call with no data, and it gives no estimate over observations that the view excluded.

The asset axis is different, and a state has a slice on it. [`port_opt_view`](@ref) keeps a subset of the assets. A family whose state has an exact sub-state over that subset returns it by an index copy.

# Arguments

  - `::AbstractPartialFitState`: The state the estimator carries, read for its type alone.
  - `::Any`: Index of the observations to keep, which no slice of a state reads.

# Returns

  - `nothing`.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`obs_weights_view`](@ref)
  - [`port_opt_view`](@ref)
  - [`partial_fit!`](@ref)
"""
obs_weights_view(::AbstractPartialFitState, ::Any) = nothing
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses two partial-fit states that cannot be merged, naming the reason.

Two states merge only when they describe the same estimate over the same assets. So the two operands must be the same struct, and each array field must have the same shape in both. The check applies to a state whose array sizes depend on the assets alone. A [`SampleBufferState`](@ref) holds one row for each observation, so its merge does not call this check.

A family whose merge needs more adds a method that calls this check first. An exponentially weighted state is one. After this check, its method refuses every pair, because the state does not record a reset of an asset inside the second block.

# Algorithm

 1. Refuse the pair when the two operands are not the same struct, whatever their type parameters.
 2. Walk the fields of the pair in declaration order.
 3. Refuse the pair when a field holds an array in both operands and the two shapes differ.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` are the same struct. The function throws an `ArgumentError` otherwise.
  - Every array field of `a` has the shape of its counterpart in `b`. The function throws a `DimensionMismatch` otherwise.

# Returns

  - `nothing`.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`merge_states`](@ref)
  - [`chan_merge`](@ref)
"""
function assert_mergeable_states(a::AbstractPartialFitState, b::AbstractPartialFitState)
    @argcheck(Base.typename(typeof(a)) === Base.typename(typeof(b)),
              ArgumentError("two partial-fit states of different types cannot be merged, but `a` is a `$(typeof(a))` and `b` is a `$(typeof(b))`."))
    for i in 1:nfields(a)
        fa = getfield(a, i)
        fb = getfield(b, i)
        if isa(fa, AbstractArray) && isa(fb, AbstractArray)
            @argcheck(size(fa) == size(fb),
                      DimensionMismatch("two partial-fit states over different numbers of assets cannot be merged, but field `$(fieldname(typeof(a), i))` has size $(size(fa)) in `a` and $(size(fb)) in `b`."))
        end
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Folds the observation count, the mean and the second-moment accumulator of two blocks into those of the concatenated block.

Equation 2.1 of [chan1979](@cite) states the merge for one series, and Equation 3.1 of [pebay2008](@cite) states it for a co-moment matrix. The merge is associative and symmetric in the two blocks. So any split of a sample into disjoint blocks gives the state of the whole sample, in any order of the merges. ``M`` is the sum of the products of the deviations, not the variance. So a caller divides it by the count, or by the count minus one, after the last merge.

# Mathematical definition

```math
\\begin{align}
n &= n_A + n_B\\\\
\\boldsymbol{\\delta} &= \\boldsymbol{\\mu}_B - \\boldsymbol{\\mu}_A\\\\
\\boldsymbol{\\mu} &= \\boldsymbol{\\mu}_A + \\boldsymbol{\\delta} \\frac{n_B}{n}\\\\
M &= M_A + M_B + \\boldsymbol{\\delta} \\boldsymbol{\\delta}^{\\intercal} \\frac{n_A n_B}{n}\\, .
\\end{align}
```

Where:

  - ``n_A``, ``n_B``: The observation counts of the two blocks.
  - ``\\boldsymbol{\\mu}_A``, ``\\boldsymbol{\\mu}_B``: The means of the two blocks.
  - ``M_A``, ``M_B``: The second-moment accumulators of the two blocks, each the sum of the products of the deviations from the mean of its block.
  - ``\\boldsymbol{\\delta}``: The difference of the two means.

When ``M`` holds one accumulator for each asset, and not a co-moment matrix, the outer product ``\\boldsymbol{\\delta} \\boldsymbol{\\delta}^{\\intercal}`` becomes its diagonal, the elementwise square ``\\boldsymbol{\\delta}^2``. For one series with the block sums ``T_A = n_A \\mu_A`` and ``T_B = n_B \\mu_B``, the last term equals ``\\frac{n_A}{n_B n} \\left( \\frac{n_B}{n_A} T_A - T_B \\right)^2``, which is the form of Equation 2.1b of [chan1979](@cite).

# Algorithm

 1. Add the two counts, giving `n`.
 2. Subtract the two means, giving `delta`.
 3. Move the first mean along `delta` by the share of the total that the second block holds, giving `mu`.
 4. Add the two accumulators, and add the outer product of `delta` scaled by `n_a * n_b / n`, giving `M`.

# Arguments

  - `n_a`: Observation count of the first block.
  - `mu_a`: Mean of the first block.
  - `M_a`: Second-moment accumulator of the first block.
  - `n_b`: Observation count of the second block.
  - `mu_b`: Mean of the second block.
  - `M_b`: Second-moment accumulator of the second block.

# Validation

  - `n_a + n_b > 0`. The function throws a `DomainError` otherwise, because the merge divides by the total count.

# Returns

  - `(n, mu, M)::Tuple`: The count, the mean and the accumulator of the concatenated block.

# Examples

```jldoctest
julia> PortfolioOptimisers.chan_merge(1, [1.0, 2.0], [0.0, 0.0], 1, [3.0, 6.0], [0.0, 0.0])
(2, [2.0, 4.0], [2.0, 8.0])
```

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`merge_states`](@ref)

# References

  - $(ref_dict[:chan1979]) Equation 2.1, page 4.
  - $(ref_dict[:pebay2008]) Equation 3.1, page 12.
"""
function chan_merge(n_a::Number, mu_a::Union{Number, VecNum}, M_a::Union{Number, VecNum},
                    n_b::Number, mu_b::Union{Number, VecNum}, M_b::Union{Number, VecNum})
    n = n_a + n_b
    @argcheck(n > zero(n), DomainError(n, "n_a + n_b must be positive"))
    delta = mu_b .- mu_a
    mu = mu_a .+ delta .* (n_b / n)
    M = M_a .+ M_b .+ delta .^ 2 .* (n_a * n_b / n)
    return n, mu, M
end
function chan_merge(n_a::Number, mu_a::VecNum, M_a::MatNum, n_b::Number, mu_b::VecNum,
                    M_b::MatNum)
    n = n_a + n_b
    @argcheck(n > zero(n), DomainError(n, "n_a + n_b must be positive"))
    delta = mu_b .- mu_a
    mu = mu_a .+ delta .* (n_b / n)
    M = M_a .+ M_b .+ (delta * transpose(delta)) .* (n_a * n_b / n)
    return n, mu, M
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Combines two partial-fit states fitted on disjoint blocks of observations into the state of the concatenated block.

Each state family implements its own method, so a call reaches this generic method only for a pair that no family answers. The method refuses a pair that cannot merge, and it names the method that the family must implement. The verb is not `Base.merge`. On a `Dict` and on a `NamedTuple`, `Base.merge` keeps the value of the right operand when a key is in both, and this merge adds the two states.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref), which names a type mismatch and a mismatch of the asset count.
 2. Throw an `ArgumentError` that names the `merge_states` method that the family of the state must implement.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).
  - The family of `a` implements `merge_states`. The function throws an `ArgumentError` otherwise.

# Returns

  - `state::AbstractPartialFitState`: The state that the fit of the two blocks as one block gives.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`assert_mergeable_states`](@ref)
  - [`chan_merge`](@ref)
"""
function merge_states(a::AbstractPartialFitState, b::AbstractPartialFitState)
    assert_mergeable_states(a, b)
    name = Base.typename(typeof(a)).name
    return throw(ArgumentError("$name is an AbstractPartialFitState with no `merge_states` method. Implement `merge_states(a::$name, b::$name)`, which calls `assert_mergeable_states` first, refuses any further mismatch of its own, and folds the two states with `chan_merge`. See the `AbstractPartialFitState` docstring for the interface."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a partial-fit state whose count, mean or accumulator cannot describe a sample.

The states of the sample mean, variance and covariance hold the same three quantities, so one check applies to the three. The count must not be negative, and the mean must be finite and not empty. Each axis of the accumulator must have the length of the mean. The accumulator holds one entry for each asset in a variance state, and it is a co-moment matrix in a covariance state. The axis rule reads both.

# Algorithm

 1. Refuse a negative `n`.
 2. Refuse an empty or non-finite `mu`.
 3. Return when `M` is `nothing`, which is the mean-only state.
 4. Refuse a non-finite `M`, and refuse an `M` whose any axis does not have the length of `mu`.

# Arguments

  - $(arg_dict[:pf_n])
  - $(arg_dict[:pf_mu])
  - `M`: The second-moment accumulator, or `nothing` for a state that holds a mean alone.

# Validation

  - `n >= 0`. The function throws a `DomainError` otherwise.
  - `!isempty(mu)`. The function throws an `IsEmptyError` otherwise.
  - Every entry of `mu` is finite. The function throws an `IsNonFiniteError` otherwise.
  - Every entry of `M` is finite. The function throws an `IsNonFiniteError` otherwise.
  - Every axis of `M` has the length of `mu`. The function throws a `DimensionMismatch` otherwise.

# Returns

  - `nothing`.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`merge_states`](@ref)
"""
function assert_partial_fit_state(n::Integer, mu::VecNum, M::Option{<:ArrNum} = nothing)
    assert_nonneg(n, :n)
    assert_nonempty(mu, :mu)
    assert_all_finite(mu, :mu)
    if !isnothing(M)
        assert_all_finite(M, :M)
        @argcheck(all(==(length(mu)), size(M)),
                  DimensionMismatch("every axis of `M` must have the length of `mu`, but `mu` has length $(length(mu)) and `M` has size $(size(M))."))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the partial-fit state that an estimator holds, and refuses an estimator that holds none.

A batch verb with the estimator alone, and no data, reads the state from the `cache` field. That field holds `nothing` until the first fold. The refusal names the verb that fills the field, so the caller gets an instruction and not a `MethodError`. An estimator with no `cache` field gets the same refusal.

# Arguments

  - `est`: The estimator whose `cache` field holds the state.

# Validation

  - `est.cache` is not `nothing`. The function throws an `ArgumentError` otherwise.

# Returns

  - `state::AbstractPartialFitState`: The state that the estimator holds.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit_cache(est::Union{AbstractEstimator, StatsBase.CovarianceEstimator})
    # An estimator with no `cache` field at all reads as one carrying nothing, so a family
    # that implements no `partial_fit!` meets the same named refusal rather than a
    # `FieldError`.
    cache = hasfield(typeof(est), :cache) ? getfield(est, :cache) : nothing
    @argcheck(!isnothing(cache),
              ArgumentError("`$(typeof(est))` carries no partial-fit state, so there is nothing to read. Call `partial_fit!` on it first, or pass a state as the second argument."))
    return cache
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the number of observations that an estimator folded into its partial-fit state.

The states of the sample moments and the sample buffer record this count in an `n` field. A call with no data that needs the sample size, and not the sample, reads the count from the state. [`Statistics.cov`](@ref) of a [`PortfolioOptimisersCovariance`](@ref) is one such call. Its matrix processing reads `size(X, 1)` alone in a batch fit, and the folded count is that number, with the rows that hold a `NaN`.

An exponentially weighted state counts the observations of each asset, and it records no count of the folded rows. The function refuses such a state with a message that names it.

# Arguments

  - `est`: The estimator whose state holds the count.

# Validation

  - `est.cache` is not `nothing`. The function throws an `ArgumentError` otherwise.
  - The state has an `n` field. The function throws an `ArgumentError` otherwise.

# Returns

  - `n::Integer`: The number of observations folded so far.

# Related

  - [`partial_fit_cache`](@ref)
  - [`partial_fit!`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function observation_count(est::Union{AbstractEstimator, StatsBase.CovarianceEstimator})
    state = partial_fit_cache(est)
    @argcheck(hasfield(typeof(state), :n),
              ArgumentError("`$(nameof(typeof(state)))` records no count of the observations that it folded, so a call with no data cannot read the sample size from it. Pass the observations, as in `cov(ce, X)`."))
    return getfield(state, :n)
end
"""
    shared_series_member_rows(X::MatNum, gaps::Nothing; kwargs...)
    shared_series_member_rows(X::MatNum, gaps::AbstractVector{Bool}; active_mask = nothing,
                              kwargs...)

Returns the observations of a carry step that its members fold, and the keyword arguments of their fold.

A carry fold adds each observation to running moments. A batch fit drops an observation with a Shared Series gap, see [`shared_series_view`](@ref), so the running moments must not get it. The carried rows keep it, because the Fold Context of an optimiser holds one row for each observation. So the members fold the unmarked rows of `X` and the same rows of the active mask. The carried rows keep every row and its mark.

# Algorithm

The method that Julia selects is the algorithm.

 1. When `gaps` is `nothing`, return `X` and the keyword arguments with no change.
 2. Otherwise, find the rows that `gaps` does not mark. Return the views of `X` and of `active_mask` at those rows, and the other keyword arguments.

# Arguments

  - `X`: The observations of the step, `observations × assets`.
  - `gaps`: One mark for each observation, `true` at an observation with a Shared Series gap, or `nothing`.
  - $(arg_dict[:pf_active_mask])
  - `kwargs...`: The other keyword arguments of the fold of a member, unchanged.

# Returns

  - `(X, kwargs)::Tuple`: The observations that the members fold, and the keyword arguments of their fold.

# Related

  - [`fold_member`](@ref)
  - [`shared_series_gaps`](@ref)
  - [`fold_prior_step`](@ref)
"""
function shared_series_member_rows(X::MatNum, ::Nothing; kwargs...)
    return X, (; kwargs...)
end
function shared_series_member_rows(X::MatNum, gaps::AbstractVector{Bool};
                                   active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                                   kwargs...)
    rows = findall(!, gaps)
    kw = if isnothing(active_mask)
        (; kwargs...)
    else
        (; active_mask = view(active_mask, rows, :), kwargs...)
    end
    return view(X, rows, :), kw
end
export partial_fit!, partial_fit
public merge_states, assert_mergeable_states, chan_merge
