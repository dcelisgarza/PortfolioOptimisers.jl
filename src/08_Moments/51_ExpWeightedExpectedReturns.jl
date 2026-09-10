"""
$(DocStringExtensions.TYPEDEF)

Estimates expected returns by an exponentially weighted recursion that freezes on a holiday and resets on an inactive period.

The recursion is seeded at zero, so a newly listed asset starts from a cold state and the output divides out the damping that the cold start costs. An asset below `min_obs` valid observations is `NaN`, and so is an asset that the active mask leaves inactive at the last observation.

Keeping a young asset investable has a cost the prior pays for it. A prior fitted with this estimator zero-fills the rows the asset was missing through [`scenario_fill`](@ref), because every consumer of a Prior Result reads its returns matrix; a scenario-based measure then reads a zero return where the asset had none and understates that asset's risk over those rows, while `mu` and `sigma` stay the estimate this recursion made from the rows it saw. The fill is silent at or below the fitting prior's own `fill_limit` field, a share of that asset's own observations, warns above it, and refuses any fill under `strict`; `fill_limit` defaults to `nothing`, and this family carries no `CoveragePolicy` to derive a limit from, so every fill is named.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpWeightedExpectedReturns(;
        decay::Number = exp2(-inv(40.0)),
        min_obs::Integer = round(Int, max(1, inv(log2(inv(decay))))),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> ExpWeightedExpectedReturns

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:decay])
  - `min_obs > 0`.

# Mathematical definition

The internal state of asset ``i`` after ``n_i`` valid observations is

```math
S_i = (1 - \\lambda) \\sum_{k=0}^{n_i - 1} \\lambda^{k} r_{i, n_i - k},
```

whose weights sum to ``1 - \\lambda^{n_i}`` rather than to one. The reported mean divides that sum out,

```math
\\hat{\\mu}_i = \\frac{S_i}{1 - \\lambda^{n_i}}.
```

Where:

  - ``\\lambda``: `decay`.
  - ``r_{i, t}``: the return of asset ``i`` at the valid observation ``t``.
  - ``n_i``: the count of valid observations of asset ``i``.

The correction is the **first** power of ``1 - \\lambda^{n_i}``. A second-moment estimator corrects by a different power, so the two forms are not interchangeable.

# Examples

```jldoctest
julia> me = ExpWeightedExpectedReturns();

julia> me.decay ≈ exp2(-inv(40.0))
true

julia> me.min_obs
40
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`ExpWeightedExpectedReturnsState`](@ref)
  - [`ExpWeightedVariance`](@ref)
  - [`ExpWeightedCovariance`](@ref)
  - [`partial_fit!`](@ref)
  - [`scenario_fill`](@ref)
  - [`EmpiricalPrior`](@ref)
"""
@concrete struct ExpWeightedExpectedReturns <: AbstractExpectedReturnsEstimator
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    """
    $(field_dict[:ew_cache])
    """
    cache
    function ExpWeightedExpectedReturns(decay::Number, min_obs::Integer,
                                        cache::Option{<:AbstractPartialFitState})
        assert_nonempty_gt0_finite_val(decay, :decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        return new{typeof(decay), typeof(min_obs), typeof(cache)}(decay, min_obs, cache)
    end
end
function ExpWeightedExpectedReturns(; decay::Number = exp2(-inv(40.0)),
                                    min_obs::Integer = round(Int,
                                                             max(1, inv(log2(inv(decay))))),
                                    cache::Option{<:AbstractPartialFitState} = nothing)::ExpWeightedExpectedReturns
    return ExpWeightedExpectedReturns(decay, min_obs, cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Internal mutable cache for the online mean update in [`ExpWeightedExpectedReturns`](@ref).

This type is an implementation detail and is not intended for direct use.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ExpWeightedExpectedReturns`](@ref)
"""
@concrete struct ExpWeightedExpectedReturnsState <: AbstractPartialFitState
    """
    $(field_dict[:ew_mu])
    """
    mu
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

Processes a single observation row (or column) to update the online mean cache.

An asset is valid when its return is finite and the active mask admits it. A valid asset takes the ordinary recursion, an active asset with a non-finite return freezes, and an asset that has just become inactive is reset to the cold state so that the bias correction restarts if it lists again.

# Arguments

  - `cache::ExpWeightedExpectedReturnsState`: Online mean computation cache (mutated).
  - `me::ExpWeightedExpectedReturns`: Expected returns estimator configuration.
  - `X::VecNum`: Returns vector for the current observation.
  - `active_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask of currently active assets. An asset that becomes inactive has its mean and count reset. With `nothing` every asset is active, so a non-finite return reads as a holiday.

# Returns

  - `cache::ExpWeightedExpectedReturnsState`: The cache to read on and to pass to the next observation. Every field is an array that is mutated in place.

# Related

  - [`ExpWeightedExpectedReturnsState`](@ref)
  - [`ExpWeightedExpectedReturns`](@ref)
"""
function process_observation!(cache::ExpWeightedExpectedReturnsState,
                              me::ExpWeightedExpectedReturns, X::VecNum,
                              active_mask::Option{<:AbstractVector{<:Bool}})
    finite_mask = isfinite.(X)
    valid = isnothing(active_mask) ? finite_mask : (finite_mask .& active_mask)

    if !isnothing(active_mask)
        newly_inactive = .!active_mask .& cache.active
        if any(newly_inactive)
            cache.mu[newly_inactive] .= zero(eltype(cache.mu))
            cache.obs_count[newly_inactive] .= 0
        end
        cache.active .= active_mask
    else
        cache.active .= true
    end

    if !any(valid)
        return cache
    end

    cache.mu[valid] .= me.decay * view(cache.mu, valid) +
                       (one(me.decay) - me.decay) * view(X, valid)
    cache.obs_count[valid] .+= 1

    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run one forward pass of an exponentially weighted update over the observations of `X`, and call `f` after each observation.

The pass owns the argument validation, the orientation and the cache, so every verb of the family states the recursion once. `f` receives the observation index and the cache as it stands after that observation.

# Arguments

  - `f`: Callback of the observation index and the cache, called after each observation.
  - `est`: Estimator of the exponentially weighted family.
  - $(arg_dict[:X])
  - `dims::Int`: Dimension along which the observations run.
  - `active_mask::Option{<:AbstractMatrix{<:Bool}}`: Optional boolean matrix with the same size as `X`.
  - `state`: Optional state to continue from. A `nothing` starts from the cold state.

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.
  - If `state` is not `nothing`, it holds as many assets as `X`.

# Returns

  - `cache`: The cache after the last observation.

# Related

  - [`ExpWeightedExpectedReturns`](@ref)
  - [`exp_weighted_moment`](@ref)
"""
function exp_weighted_pass!(f, est::ExpWeightedExpectedReturns, X::MatNum, dims::Int,
                            active_mask::Option{<:AbstractMatrix{<:Bool}},
                            state::Option{<:ExpWeightedExpectedReturnsState} = nothing)
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
        ExpWeightedExpectedReturnsState(zeros(eltype(X), N), zeros(Int, N), trues(N))
    else
        @argcheck(length(state.mu) == N,
                  DimensionMismatch("the state holds $(length(state.mu)) assets, and `X` holds $N"))
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

Run one forward pass of an exponentially weighted update over the observations of `X`, and read no intermediate cache.

This is the callback method with a callback that does nothing, so a verb that wants the last cache alone states no callback of its own. The callback method owns the recursion, the argument validation and the cache.

# Arguments

  - `est`: Estimator of the exponentially weighted family.
  - $(arg_dict[:X])
  - `dims::Int`: Dimension along which the observations run.
  - `active_mask::Option{<:AbstractMatrix{<:Bool}}`: Optional boolean matrix with the same size as `X`.
  - `state`: Optional state to continue from. A `nothing` starts from the cold state.

# Returns

  - `cache`: The cache after the last observation.

# Related

  - [`ExpWeightedExpectedReturns`](@ref)
  - [`exp_weighted_moment`](@ref)
"""
function exp_weighted_pass!(est::ExpWeightedExpectedReturns, X::MatNum, dims::Int,
                            active_mask::Option{<:AbstractMatrix{<:Bool}},
                            state::Option{<:ExpWeightedExpectedReturnsState} = nothing)
    return exp_weighted_pass!((args...) -> nothing, est, X, dims, active_mask, state)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Read the exponentially weighted mean out of a cache, as it stands.

Applies the cold-start bias correction and blanks every asset that is not ready. The cache is read, never written, so the same cache answers this call after every observation of a forward pass.

# Arguments

  - `cache::ExpWeightedExpectedReturnsState`: Online mean computation cache.
  - `est::ExpWeightedExpectedReturns`: Expected returns estimator configuration.

# Returns

  - `mu::Vector{<:Number}`: Per-asset expected returns vector. An asset that is inactive, or that carries fewer than `est.min_obs` valid observations, is `NaN`.

# Related

  - [`ExpWeightedExpectedReturnsState`](@ref)
  - [`ExpWeightedExpectedReturns`](@ref)
  - [`exp_weighted_pass!`](@ref)
"""
function exp_weighted_moment(cache::ExpWeightedExpectedReturnsState,
                             est::ExpWeightedExpectedReturns)
    mu = copy(cache.mu)
    counted = cache.obs_count .> zero(eltype(cache.obs_count))
    correction = ones(eltype(mu), length(mu))
    correction[counted] .= inv.(max.(one(est.decay) .-
                                     est.decay .^ view(cache.obs_count, counted),
                                     eps(eltype(mu))))
    mu .*= correction
    not_ready = .!cache.active .| (cache.obs_count .< est.min_obs)
    if any(not_ready)
        mu[not_ready] .= NaN
    end

    return mu
end
"""
    Statistics.mean(
        me::ExpWeightedExpectedReturns,
        X::MatNum;
        dims::Int = 1,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Vector{<:Number}

Compute the exponentially weighted expected returns of each asset.

Iterates over the observation dimension of `X`, updating an online mean cache at each step, then applies the cold-start bias correction and blanks every asset that is not ready.

# Arguments

  - `me`: Exponentially weighted expected returns estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`. An asset whose entry is `false` is inactive at that observation: its state is reset and its answer is `NaN` while it stays inactive. With `nothing` every asset is active, so a non-finite return reads as a holiday and the mean freezes.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `mu::Vector{<:Number}`: Per-asset expected returns vector of length `assets`. An asset with fewer than `me.min_obs` valid observations is `NaN`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> me = ExpWeightedExpectedReturns(; decay = 0.9, min_obs = 2);

julia> length(mean(me, X))
2
```

# Related

  - [`ExpWeightedExpectedReturns`](@ref)
  - [`ExpWeightedExpectedReturnsState`](@ref)
  - [`exp_weighted_moment`](@ref)
"""
function Statistics.mean(me::ExpWeightedExpectedReturns, X::MatNum; dims::Int = 1,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = exp_weighted_pass!(me, X, dims, active_mask)
    return exp_weighted_moment(cache, me)
end
"""
    Statistics.mean(
        me::ExpWeightedExpectedReturns,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> Vector{<:Number}

Compute the exponentially weighted expected returns from a window of an Asset Panel.

This estimator is mask-aware, so it overrides the reduce-and-expand root of the verb and reads the panel's active mask itself. The answer therefore lives on the whole universe rather than on the Coverage Universe: a young asset that lists inside the window is answered from the observations it has, and it is `NaN` only while it stays below `me.min_obs`.

# Arguments

  - `me`: Exponentially weighted expected returns estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `mu::Vector{<:Number}`: Per-asset expected returns vector of length `assets`.

# Related

  - [`ExpWeightedExpectedReturns`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`Statistics.mean(me::AbstractExpectedReturnsEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.mean(me::ExpWeightedExpectedReturns, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, _ = panel_moment_masks(pnl)
    return Statistics.mean(me, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    partial_fit!(
        me::ExpWeightedExpectedReturns,
        X::MatNum;
        dims::Int = 1,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> ExpWeightedExpectedReturns

Fold a block of observations into the estimator's own online mean state.

The estimator carries the state in its `cache` field, so a second call continues the recursion rather than restarting it. This is the exception ADR 0106 states: a partial-fit state is the one Result an estimator may hold.

# Arguments

  - `me`: Exponentially weighted expected returns estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.
  - If `me.cache` is not `nothing`, it holds as many assets as `X`.

# Returns

  - `me::ExpWeightedExpectedReturns`: The estimator, with `cache` holding the state after the block.

# Related

  - [`ExpWeightedExpectedReturns`](@ref)
  - [`ExpWeightedExpectedReturnsState`](@ref)
  - [`exp_weighted_pass!`](@ref)
  - [`Statistics.mean(me::ExpWeightedExpectedReturns; kwargs...)`](@ref)
"""
function partial_fit!(me::ExpWeightedExpectedReturns{<:Any, <:Any,
                                                     <:Option{<:ExpWeightedExpectedReturnsState}},
                      X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = exp_weighted_pass!(me, X, dims, active_mask, me.cache)
    Accessors.@reset me.cache = cache
    return me
end
"""
    partial_fit!(
        me::ExpWeightedExpectedReturns,
        x::VecNum;
        active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
        kwargs...
    ) -> ExpWeightedExpectedReturns

Fold one observation into the estimator's own online mean state.

The entries of `x` are the assets of a single observation, which is the row the matrix method folds one at a time. This is the shape a caller has when the observations arrive one by one.

# Arguments

  - `me`: Exponentially weighted expected returns estimator.
  - `x::VecNum`: One observation, with one entry per asset.
  - `active_mask`: Optional boolean vector with the same length as `x`.
  - $(arg_dict[:ignkwargs])

# Validation

  - If `active_mask` is not `nothing`, `length(x) == length(active_mask)`.
  - If `me.cache` is not `nothing`, it holds as many assets as `x`.

# Returns

  - `me::ExpWeightedExpectedReturns`: The estimator, with `cache` holding the state after the observation.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> me = ExpWeightedExpectedReturns(; decay = 0.9, min_obs = 2);

julia> one_at_a_time = foldl((c, i) -> partial_fit!(c, view(X, i, :)), axes(X, 1); init = me);

julia> isequal(mean(one_at_a_time), mean(me, X))
true
```

# Related

  - [`partial_fit!(me::ExpWeightedExpectedReturns, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`ExpWeightedExpectedReturns`](@ref)
"""
function partial_fit!(me::ExpWeightedExpectedReturns{<:Any, <:Any,
                                                     <:Option{<:ExpWeightedExpectedReturnsState}},
                      x::VecNum; active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      kwargs...)
    return partial_fit!(me, permutedims(x); dims = 1,
                        active_mask = if isnothing(active_mask)
                            nothing
                        else
                            permutedims(active_mask)
                        end)
end
"""
    Statistics.mean(
        me::ExpWeightedExpectedReturns,
        state::ExpWeightedExpectedReturnsState;
        kwargs...
    ) -> Vector{<:Number}

Read the exponentially weighted expected returns out of a state the caller holds.

# Arguments

  - `me`: Exponentially weighted expected returns estimator.
  - `state::ExpWeightedExpectedReturnsState`: The state to read.
  - $(arg_dict[:ignkwargs])

# Returns

  - `mu::Vector{<:Number}`: Per-asset expected returns vector.

# Related

  - [`ExpWeightedExpectedReturnsState`](@ref)
  - [`exp_weighted_moment`](@ref)
  - [`Statistics.mean(me::ExpWeightedExpectedReturns; kwargs...)`](@ref)
"""
function Statistics.mean(me::ExpWeightedExpectedReturns,
                         state::ExpWeightedExpectedReturnsState; kwargs...)
    return exp_weighted_moment(state, me)
end
"""
    Statistics.mean(me::ExpWeightedExpectedReturns; kwargs...) -> Vector{<:Number}

Read the exponentially weighted expected returns out of the estimator's own state.

The one-argument form is what an incremental fit answers: [`partial_fit!`](@ref) leaves the state in the `cache` field, and this verb turns it into the ordinary answer. An estimator that has been given no observation carries no state, so the call is refused rather than answered with a zero.

# Arguments

  - `me`: Exponentially weighted expected returns estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `me.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `mu::Vector{<:Number}`: Per-asset expected returns vector of length `assets`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> me = partial_fit!(ExpWeightedExpectedReturns(; decay = 0.9, min_obs = 2), X);

julia> length(mean(me))
2

julia> mean(ExpWeightedExpectedReturns())
ERROR: ArgumentError: `me` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(me, X)` first, or `mean(me, X)` for a fit over a whole sample.
[...]
```

# Related

  - [`partial_fit!(me::ExpWeightedExpectedReturns, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`Statistics.mean(me::ExpWeightedExpectedReturns, state::ExpWeightedExpectedReturnsState; kwargs...)`](@ref)
  - [`ExpWeightedExpectedReturnsState`](@ref)
"""
function Statistics.mean(me::ExpWeightedExpectedReturns; kwargs...)
    state = me.cache
    @argcheck(!isnothing(state),
              ArgumentError("`me` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(me, X)` first, or `mean(me, X)` for a fit over a whole sample."))
    return mean(me, state)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`ExpWeightedExpectedReturnsState`](@ref).

An exponentially weighted state folds forward exactly, `S = λ^{n_b} S_a + S_b`, but only while no asset resets inside the second block. The state records the count that a reset zeroed and not the reset itself, so the two cases are indistinguishable after the fact and a merge would silently keep a history the reset discarded. Fold the second block into the first with `partial_fit!` instead.

# Arguments

  - `a`: State of the first block.
  - `b`: State of the second block.

# Validation

  - The pair is refused with an `ArgumentError`.

# Related

  - [`ExpWeightedExpectedReturnsState`](@ref)
  - [`merge_states`](@ref)
  - [`assert_mergeable_states`](@ref)
"""
function merge_states(a::ExpWeightedExpectedReturnsState,
                      b::ExpWeightedExpectedReturnsState)
    assert_mergeable_states(a, b)
    return throw(ArgumentError("an `ExpWeightedExpectedReturnsState` pair does not merge, because the state does not record whether an asset reset inside the second block. A reset zeroes the count that the fold would read, so the two histories are indistinguishable. Fold the second block into the first with `partial_fit!` instead."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`ExpWeightedExpectedReturnsState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. Every field is an array, and every one is copied.

# Arguments

  - `x`: The cache to copy.

# Returns

  - `state::ExpWeightedExpectedReturnsState`: A fresh cache, equal to `x`, whose arrays are fresh.

# Related

  - [`ExpWeightedExpectedReturnsState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::ExpWeightedExpectedReturnsState)
    return ExpWeightedExpectedReturnsState(copy(x.mu), copy(x.obs_count), copy(x.active))
end

export ExpWeightedExpectedReturns
