"""
    partial_fit!(est, X)

Folds observations into an estimator's partial-fit state, and returns the estimator.

An incremental fit reads each observation once and keeps what it needs in an [`AbstractPartialFitState`](@ref), so a later call continues where the last one stopped instead of reading the sample again. The state lives in the estimator's `cache` field, which holds `nothing` until the first call. ADR 0106 records why that field is the one Result an estimator holds.

The verb mutates the array fields of the state and rebinds its scalar fields with `Accessors.@reset`, so it returns a **new** estimator and the caller must rebind it. The returned estimator shares the state object with the one it was built from, which is what the `!` in the name says: two estimators returned by successive calls read the same running quantities.

A batch verb ignores the state. `var(ce, X)` fits `X` alone, so an estimator carrying a state still answers any input it is given.

# Interfaces

A family that answers this verb implements two methods:

  - `partial_fit!(est, X::MatNum; dims::Int = 1, kwargs...) -> est`: Folds every observation of `X`, in order.
  - `partial_fit!(est, x::VecNum; kwargs...) -> est`: Folds one observation, whose entries are the assets.

# Arguments

  - `est`: Estimator whose state is folded forward.
  - `X`: Observations to fold. A matrix holds one observation per row when `dims == 1`, and one per column when `dims == 2`. A vector is a single observation across the assets.

# Returns

  - `est`: The estimator, with its `cache` field rebound to the state after the last observation.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`merge_states`](@ref)
"""
function partial_fit! end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses two partial-fit states that cannot be merged, naming the reason.

Two states merge only when they describe the same estimate over the same assets, so this check holds for every state family: the two operands are the same struct, and every array field they hold has the same shape. A family whose merge needs more than that adds a method of its own that calls this one first. An exponentially weighted pair is the case that needs it, because two states of different decay weight the same observation differently.

# Algorithm

 1. Refuse the pair when the two operands are not the same struct, whatever their type parameters.
 2. Walk the fields of the pair in declaration order.
 3. Refuse the pair when a field holds an array in both operands and the two shapes differ.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` are the same struct. An `ArgumentError` is thrown otherwise.
  - Every array field of `a` has the shape of its counterpart in `b`. A `DimensionMismatch` is thrown otherwise.

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

This is the merge of [chan1983](@cite), and it is what makes an incremental fit parallel and associative: a sample split into any set of disjoint blocks gives the state of the whole sample, whatever order the blocks are folded in. The second moment is the accumulator ``M``, not the variance, so a caller divides by the count or by the count less one after the last merge.

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

  - ``n_A``, ``n_B``: observation counts of the two blocks.
  - ``\\boldsymbol{\\mu}_A``, ``\\boldsymbol{\\mu}_B``: means of the two blocks.
  - ``M_A``, ``M_B``: second-moment accumulators of the two blocks.
  - ``\\boldsymbol{\\delta}``: difference of the two means.

The outer product ``\\boldsymbol{\\delta} \\boldsymbol{\\delta}^{\\intercal}`` becomes the elementwise square ``\\boldsymbol{\\delta}^2`` when ``M`` is per-asset rather than a co-moment matrix.

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

  - `n_a + n_b > 0`. A `DomainError` is thrown otherwise, because the merge divides by the total count.

# Returns

  - `(n, mu, M)::Tuple`: The count, the mean and the accumulator of the concatenated block.

# Examples

```jldoctest
julia> PortfolioOptimisers.chan_merge(1, [1.0, 2.0], [0.0, 0.0], 1, [3.0, 6.0], [0.0, 0.0])
(2, [2.0, 4.0], [2.0, 8.0])
```

# References

  - $(ref_dict[:chan1983])

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`merge_states`](@ref)
"""
function chan_merge(n_a::Number, mu_a::Union{<:Number, <:VecNum},
                    M_a::Union{<:Number, <:VecNum}, n_b::Number,
                    mu_b::Union{<:Number, <:VecNum}, M_b::Union{<:Number, <:VecNum})
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

Each state family implements its own method, so this generic method is reached only by a pair no family answers. It refuses the pair that cannot merge at all, and names the method the family still owes. The verb is deliberately not `Base.merge`, whose contract on a `Dict` and on a `NamedTuple` is that the right operand wins a key conflict, where this merge is a sum.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref), which names a type mismatch and an asset-count mismatch.
 2. Throw an `ArgumentError` naming the `merge_states` method the state's own family must implement.

# Arguments

  - `a`: The state of the first block of observations.
  - `b`: The state of the second block of observations.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).
  - The family of `a` implements `merge_states`. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::AbstractPartialFitState`: The state the two blocks give when they are fitted as one block.

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

Every second-order state carries the same three quantities, so one check serves the whole family: a non-negative count, a non-empty finite mean, and an accumulator whose every axis has the length of the mean. The accumulator is per-asset in a variance state and a co-moment matrix in a covariance state, and the axis rule reads both.

# Algorithm

 1. Refuse a negative `n`.
 2. Refuse an empty or non-finite `mu`.
 3. Return when `M` is `nothing`, which is the mean-only state.
 4. Refuse a non-finite `M`, and refuse an `M` whose any axis does not have the length of `mu`.

# Arguments

  - $(arg_dict[:pf_n])
  - $(arg_dict[:pf_mu])
  - `M`: The second-moment accumulator, or `nothing` for a state that carries a mean alone.

# Validation

  - `n >= 0`. A `DomainError` is thrown otherwise.
  - `!isempty(mu)`. An `IsEmptyError` is thrown otherwise.
  - Every entry of `mu` is finite. An `IsNonFiniteError` is thrown otherwise.
  - Every entry of `M` is finite. An `IsNonFiniteError` is thrown otherwise.
  - Every axis of `M` has the length of `mu`. A `DimensionMismatch` is thrown otherwise.

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

Returns the partial-fit state an estimator carries, and refuses an estimator that carries none.

A read-out verb called on the estimator alone reads the state out of the `cache` field, and that field holds `nothing` until the first fold. The refusal names the verb that fills it, so a caller who reached the one-argument form too early is told what to call rather than meeting a `MethodError`.

# Arguments

  - `est`: The estimator whose `cache` field carries the state.

# Validation

  - `est.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `state::AbstractPartialFitState`: The state the estimator carries.

# Related

  - [`AbstractPartialFitState`](@ref)
  - [`partial_fit!`](@ref)
"""
function partial_fit_cache(est::Union{<:AbstractEstimator, <:StatsBase.CovarianceEstimator})
    cache = est.cache
    @argcheck(!isnothing(cache),
              ArgumentError("`$(typeof(est))` carries no partial-fit state, so there is nothing to read. Call `partial_fit!` on it first, or pass a state as the second argument."))
    return cache
end

export partial_fit!
