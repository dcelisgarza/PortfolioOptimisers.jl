"""
$(DocStringExtensions.TYPEDEF)

Estimates per-asset variance by an exponentially weighted recursion that freezes on a holiday and resets on an inactive period.

The recursion is seeded at zero, so a newly listed asset starts from a cold state and the output divides out the damping that the cold start costs. An asset below `min_obs` valid observations is `NaN`, and so is an asset that the active mask leaves inactive at the last observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpWeightedVariance(;
        decay::Number = exp2(-inv(40.0)),
        min_obs::Integer = round(Int, max(1, inv(log2(inv(decay))))),
        centred::Bool = false,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> ExpWeightedVariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:decay])
  - `min_obs > 0`.

# Mathematical definition

The internal state of asset ``i`` after ``n_i`` valid observations is

```math
S_i = (1 - \\lambda) \\sum_{k=0}^{n_i - 1} \\lambda^{k} e_{i, n_i - k}^{2},
```

and the reported variance divides out the weights the cold start never accumulated,

```math
\\hat{\\sigma}^{2}_i = \\frac{S_i}{1 - \\lambda^{n_i}}.
```

Where:

  - ``\\lambda``: `decay`.
  - ``e_{i, t}``: the return of asset ``i`` at the valid observation ``t``, less the running location where `centred` is `false`, and the return itself where it is `true`.
  - ``n_i``: the count of valid observations of asset ``i``.

# Examples

```jldoctest
julia> ce = ExpWeightedVariance();

julia> ce.decay ≈ exp2(-inv(40.0))
true

julia> ce.min_obs
40
```

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`ExpWeightedVarianceState`](@ref)
  - [`ExpWeightedExpectedReturns`](@ref)
  - [`ExpWeightedCovariance`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`partial_fit!`](@ref)
"""
@concrete struct ExpWeightedVariance <: AbstractVarianceEstimator
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    """
    $(field_dict[:centred])
    """
    centred
    """
    $(field_dict[:ew_cache])
    """
    cache
    function ExpWeightedVariance(decay::Number, min_obs::Integer, centred::Bool,
                                 cache::Option{<:AbstractPartialFitState})
        assert_nonempty_gt0_finite_val(decay, :decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        return new{typeof(decay), typeof(min_obs), typeof(centred), typeof(cache)}(decay,
                                                                                   min_obs,
                                                                                   centred,
                                                                                   cache)
    end
end
function ExpWeightedVariance(; decay::Number = exp2(-inv(40.0)),
                             min_obs::Integer = round(Int, max(1, inv(log2(inv(decay))))),
                             centred::Bool = false,
                             cache::Option{<:AbstractPartialFitState} = nothing)::ExpWeightedVariance
    return ExpWeightedVariance(decay, min_obs, centred, cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Internal mutable cache for the online variance update in [`ExpWeightedVariance`](@ref).

This type is an implementation detail and is not intended for direct use.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ExpWeightedVariance`](@ref)
"""
@concrete struct ExpWeightedVarianceState <: AbstractPartialFitState
    """
    $(field_dict[:ra_variance])
    """
    variance
    """
    $(field_dict[:ra_location])
    """
    location
    """
    $(field_dict[:obs_count])
    """
    obs_count
    """
    $(field_dict[:ra_active])
    """
    active
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Processes a single observation row (or column) to update the online variance cache.

An asset is valid when its return is finite and the active mask admits it. A valid asset takes the ordinary recursion, an active asset with a non-finite return freezes, and an asset that has just become inactive is reset to the cold state so that the bias correction restarts if it lists again. Where `centred` is `false` the deviation is taken from the location that stands before the observation, and the location is advanced afterwards.

# Arguments

  - `cache::ExpWeightedVarianceState`: Online variance computation cache (mutated).
  - `ce::ExpWeightedVariance`: Variance estimator configuration.
  - `X::VecNum`: Returns vector for the current observation.
  - `active_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask of currently active assets. An asset that becomes inactive has its variance, its location and its count reset. With `nothing` every asset is active, so a non-finite return reads as a holiday.

# Returns

  - `cache::ExpWeightedVarianceState`: The cache to read on and to pass to the next observation. Every field is an array that is mutated in place.

# Related

  - [`ExpWeightedVarianceState`](@ref)
  - [`ExpWeightedVariance`](@ref)
"""
function process_observation!(cache::ExpWeightedVarianceState, ce::ExpWeightedVariance,
                              X::VecNum, active_mask::Option{<:AbstractVector{<:Bool}})
    finite_mask = isfinite.(X)
    valid = isnothing(active_mask) ? finite_mask : (finite_mask .& active_mask)

    if !isnothing(active_mask)
        newly_inactive = .!active_mask .& cache.active
        if any(newly_inactive)
            cache.variance[newly_inactive] .= zero(eltype(cache.variance))
            cache.obs_count[newly_inactive] .= 0
            if !ce.centred
                cache.location[newly_inactive] .= NaN
            end
        end
        cache.active .= active_mask
    else
        cache.active .= true
    end

    if !any(valid)
        return cache
    end

    Xi = if ce.centred
        X
    else
        loc = replace(cache.location, NaN => zero(eltype(cache.location)))
        cache.location[valid] = ce.decay * view(loc, valid) +
                                (one(ce.decay) - ce.decay) * view(X, valid)
        X - loc
    end

    cache.variance[valid] .= ce.decay * view(cache.variance, valid) +
                             (one(ce.decay) - ce.decay) * view(Xi, valid) .^ 2
    cache.obs_count[valid] .+= 1

    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Variance method of [`exp_weighted_pass!`](@ref). Runs one forward pass of the online variance update over the observations of `X`, and calls `f` after each observation.

# Related

  - [`ExpWeightedVariance`](@ref)
  - [`ExpWeightedVarianceState`](@ref)
  - [`exp_weighted_pass!`](@ref)
"""
function exp_weighted_pass!(f, est::ExpWeightedVariance, X::MatNum, dims::Int,
                            active_mask::Option{<:AbstractMatrix{<:Bool}},
                            state::Option{<:ExpWeightedVarianceState} = nothing)
    assert_dims(dims)
    act_flag = !isnothing(active_mask)
    itr, v = ifelse(isone(dims), (eachrow, (x, y) -> view(x, y, :)),
                    (eachcol, (x, y) -> view(x, :, y)))
    if act_flag
        @argcheck(size(X) == size(active_mask),
                  DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(active_mask)))"))
    end
    N = size(X, setdiff((1, 2), (dims,))[1])

    cache = if isnothing(state)
        ExpWeightedVarianceState(zeros(eltype(X), N),
                                 est.centred ? zeros(eltype(X), N) : fill(NaN, N),
                                 zeros(Int, N), trues(N))
    else
        @argcheck(length(state.variance) == N,
                  DimensionMismatch("the state holds $(length(state.variance)) assets, and `X` holds $N"))
        state
    end
    for (i, Xi) in enumerate(itr(X))
        ami = act_flag ? v(active_mask, i) : nothing
        cache = process_observation!(cache, est, Xi, ami)
        f(i, cache)
    end

    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Variance method of [`exp_weighted_moment`](@ref). Reads the exponentially weighted variance out of a cache, as it stands.

Applies the cold-start bias correction and blanks every asset that is not ready. The cache is read, never written, so the same cache answers this call after every observation of a forward pass.

# Returns

  - `variance::Vector{<:Number}`: Per-asset variance vector. An asset that is inactive, or that carries fewer than `est.min_obs` valid observations, is `NaN`.

# Related

  - [`ExpWeightedVarianceState`](@ref)
  - [`ExpWeightedVariance`](@ref)
  - [`exp_weighted_moment`](@ref)
"""
function exp_weighted_moment(cache::ExpWeightedVarianceState, est::ExpWeightedVariance)
    variance = copy(cache.variance)
    counted = cache.obs_count .> zero(eltype(cache.obs_count))
    correction = ones(eltype(variance), length(variance))
    correction[counted] .= inv.(max.(one(est.decay) .-
                                     est.decay .^ view(cache.obs_count, counted),
                                     eps(eltype(variance))))
    variance .*= correction
    not_ready = .!cache.active .| (cache.obs_count .< est.min_obs)
    if any(not_ready)
        variance[not_ready] .= NaN
    end

    return variance
end
"""
    Statistics.var(
        ce::ExpWeightedVariance,
        X::MatNum;
        dims::Int = 1,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Vector{<:Number}

Compute the exponentially weighted variance of each asset.

Iterates over the observation dimension of `X`, updating an online variance cache at each step, then applies the cold-start bias correction and blanks every asset that is not ready.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`. An asset whose entry is `false` is inactive at that observation: its state is reset and its answer is `NaN` while it stays inactive. With `nothing` every asset is active, so a non-finite return reads as a holiday and the variance freezes.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `var::Vector{<:Number}`: Per-asset variance vector of length `assets`. An asset with fewer than `ce.min_obs` valid observations is `NaN`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = ExpWeightedVariance(; decay = 0.9, min_obs = 2);

julia> length(var(ce, X))
2
```

# Related

  - [`ExpWeightedVariance`](@ref)
  - [`ExpWeightedVarianceState`](@ref)
  - [`Statistics.std(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function Statistics.var(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = exp_weighted_pass!((args...) -> nothing, ce, X, dims, active_mask)
    if !ce.centred && any(.!cache.active)
        cache.location[.!cache.active] .= NaN
    end

    return exp_weighted_moment(cache, ce)
end
"""
    Statistics.std(
        ce::ExpWeightedVariance,
        X::MatNum;
        dims::Int = 1,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Vector{<:Number}

Compute the exponentially weighted volatility of each asset.

This is the square root of the variance of the same call.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`.
  - $(arg_dict[:ignkwargs])

# Returns

  - `std::Vector{<:Number}`: Per-asset volatility vector of length `assets`.

# Related

  - [`ExpWeightedVariance`](@ref)
  - [`Statistics.var(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function Statistics.std(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    return sqrt.(var(ce, X; dims = dims, active_mask = active_mask, kwargs...))
end
"""
    Statistics.var(
        ce::ExpWeightedVariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> Vector{<:Number}

Compute the exponentially weighted variance from a window of an Asset Panel.

This estimator is mask-aware, so it overrides the reduce-and-expand root of the verb and reads the panel's active mask itself. The answer therefore lives on the whole universe rather than on the Coverage Universe: a young asset that lists inside the window is answered from the observations it has, and it is `NaN` only while it stays below `ce.min_obs`.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `var::Vector{<:Number}`: Per-asset variance vector of length `assets`.

# Related

  - [`ExpWeightedVariance`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`Statistics.var(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.var(ce::ExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    amsk, _ = panel_moment_masks(pnl)
    return Statistics.var(ce, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    Statistics.std(
        ce::ExpWeightedVariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> Vector{<:Number}

Compute the exponentially weighted volatility from a window of an Asset Panel.

This is the square root of the variance of the same call, and it reads the panel's active mask through the same override.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `std::Vector{<:Number}`: Per-asset volatility vector of length `assets`.

# Related

  - [`ExpWeightedVariance`](@ref)
  - [`Statistics.var(ce::ExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.std(ce::ExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    amsk, _ = panel_moment_masks(pnl)
    return Statistics.std(ce, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    partial_fit!(
        ce::ExpWeightedVariance,
        X::MatNum;
        dims::Int = 1,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> ExpWeightedVariance

Fold a block of observations into the estimator's own online variance state.

The estimator carries the state in its `cache` field, so a second call continues the recursion rather than restarting it. This is the exception ADR 0106 states: a partial-fit state is the one Result an estimator may hold.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `X`.

# Returns

  - `ce::ExpWeightedVariance`: The estimator, with `cache` holding the state after the block.

# Related

  - [`ExpWeightedVariance`](@ref)
  - [`ExpWeightedVarianceState`](@ref)
  - [`Statistics.var(ce::ExpWeightedVariance; kwargs...)`](@ref)
"""
function partial_fit!(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = exp_weighted_pass!((args...) -> nothing, ce, X, dims, active_mask, ce.cache)
    Accessors.@reset ce.cache = cache
    return ce
end
"""
    partial_fit!(
        ce::ExpWeightedVariance,
        x::VecNum;
        active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
        kwargs...
    ) -> ExpWeightedVariance

Fold one observation into the estimator's own online variance state.

The entries of `x` are the assets of a single observation, which is the row the matrix method folds one at a time. This is the shape a caller has when the observations arrive one by one.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - `x::VecNum`: One observation, with one entry per asset.
  - `active_mask`: Optional boolean vector with the same length as `x`.
  - $(arg_dict[:ignkwargs])

# Validation

  - If `active_mask` is not `nothing`, `length(x) == length(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `x`.

# Returns

  - `ce::ExpWeightedVariance`: The estimator, with `cache` holding the state after the observation.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = ExpWeightedVariance(; decay = 0.9, min_obs = 2);

julia> one_at_a_time = foldl((c, i) -> partial_fit!(c, view(X, i, :)), axes(X, 1); init = ce);

julia> isequal(var(one_at_a_time), var(ce, X))
true
```

# Related

  - [`partial_fit!(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`ExpWeightedVariance`](@ref)
"""
function partial_fit!(ce::ExpWeightedVariance, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
    return partial_fit!(ce, permutedims(x); dims = 1,
                        active_mask = if isnothing(active_mask)
                            nothing
                        else
                            permutedims(active_mask)
                        end)
end
"""
    Statistics.var(ce::ExpWeightedVariance, state::ExpWeightedVarianceState; kwargs...) -> Vector{<:Number}

Read the exponentially weighted variance out of a state the caller holds.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - `state::ExpWeightedVarianceState`: The state to read.
  - $(arg_dict[:ignkwargs])

# Returns

  - `var::Vector{<:Number}`: Per-asset variance vector.

# Related

  - [`ExpWeightedVarianceState`](@ref)
  - [`exp_weighted_moment`](@ref)
"""
function Statistics.var(ce::ExpWeightedVariance, state::ExpWeightedVarianceState; kwargs...)
    return exp_weighted_moment(state, ce)
end
"""
    Statistics.var(ce::ExpWeightedVariance; kwargs...) -> Vector{<:Number}

Read the exponentially weighted variance out of the estimator's own state.

The one-argument form is what an incremental fit answers: [`partial_fit!`](@ref) leaves the state in the `cache` field, and this verb turns it into the ordinary answer. An estimator that has been given no observation carries no state, so the call is refused rather than answered with a zero.

# Arguments

  - `ce`: Exponentially weighted variance estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `var::Vector{<:Number}`: Per-asset variance vector of length `assets`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(ExpWeightedVariance(; decay = 0.9, min_obs = 2), X);

julia> length(var(ce))
2

julia> var(ExpWeightedVariance())
ERROR: ArgumentError: `ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `var(ce, X)` for a fit over a whole sample.
[...]
```

# Related

  - [`partial_fit!(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`Statistics.var(ce::ExpWeightedVariance, state::ExpWeightedVarianceState; kwargs...)`](@ref)
"""
function Statistics.var(ce::ExpWeightedVariance; kwargs...)
    state = ce.cache
    @argcheck(!isnothing(state),
              ArgumentError("`ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `var(ce, X)` for a fit over a whole sample."))
    return var(ce, state)
end
"""
    Statistics.std(ce::ExpWeightedVariance, state::ExpWeightedVarianceState; kwargs...) -> Vector{<:Number}

Read the exponentially weighted volatility out of a state the caller holds.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - `state::ExpWeightedVarianceState`: The state to read.
  - $(arg_dict[:ignkwargs])

# Returns

  - `std::Vector{<:Number}`: Per-asset volatility vector.

# Related

  - [`ExpWeightedVarianceState`](@ref)
  - [`Statistics.var(ce::ExpWeightedVariance, state::ExpWeightedVarianceState; kwargs...)`](@ref)
"""
function Statistics.std(ce::ExpWeightedVariance, state::ExpWeightedVarianceState; kwargs...)
    return sqrt.(var(ce, state; kwargs...))
end
"""
    Statistics.std(ce::ExpWeightedVariance; kwargs...) -> Vector{<:Number}

Read the exponentially weighted volatility out of the estimator's own state.

# Arguments

  - `ce`: Exponentially weighted variance estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `std::Vector{<:Number}`: Per-asset volatility vector of length `assets`.

# Related

  - [`Statistics.var(ce::ExpWeightedVariance; kwargs...)`](@ref)
  - [`ExpWeightedVarianceState`](@ref)
"""
function Statistics.std(ce::ExpWeightedVariance; kwargs...)
    return sqrt.(var(ce; kwargs...))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`ExpWeightedVarianceState`](@ref).

An exponentially weighted state folds forward exactly, `S = λ^{n_b} S_a + S_b`, but only while no asset resets inside the second block. The state records the count that a reset zeroed and not the reset itself, so the two cases are indistinguishable after the fact and a merge would silently keep a history the reset discarded. Fold the second block into the first with `partial_fit!` instead.

# Arguments

  - `a`: State of the first block.
  - `b`: State of the second block.

# Validation

  - The pair is refused with an `ArgumentError`.

# Related

  - [`ExpWeightedVarianceState`](@ref)
  - [`merge_states`](@ref)
  - [`assert_mergeable_states`](@ref)
"""
function merge_states(a::ExpWeightedVarianceState, b::ExpWeightedVarianceState)
    assert_mergeable_states(a, b)
    return throw(ArgumentError("an `ExpWeightedVarianceState` pair does not merge, because the state does not record whether an asset reset inside the second block. A reset zeroes the count that the fold would read, so the two histories are indistinguishable. Fold the second block into the first with `partial_fit!` instead."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`ExpWeightedVarianceState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. Every field is an array, and every one is copied.

# Arguments

  - `x`: The cache to copy.

# Returns

  - `state::ExpWeightedVarianceState`: A fresh cache, equal to `x`, whose arrays are fresh.

# Related

  - [`ExpWeightedVarianceState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::ExpWeightedVarianceState)
    return ExpWeightedVarianceState(copy(x.variance), copy(x.location), copy(x.obs_count),
                                    copy(x.active))
end

export ExpWeightedVariance
