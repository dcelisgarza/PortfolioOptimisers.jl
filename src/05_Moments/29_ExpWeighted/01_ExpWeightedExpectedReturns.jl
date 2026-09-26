"""
$(DocStringExtensions.TYPEDEF)

Estimates expected returns by an exponentially weighted recursion that freezes on a holiday and resets on an inactive period.

The state of each asset starts at zero, so a newly listed asset starts cold, and the answer divides out the weight that the zero start removes. An asset with fewer than `min_obs` valid observations is `NaN`, and so is an asset that is inactive at the last observation.

A young asset stays investable, and the prior pays a cost for it. Every consumer of a Prior Result reads its returns matrix, so a prior fitted with this estimator fills the rows that the asset misses with zero through [`scenario_fill`](@ref). A scenario-based risk measure then reads a zero return where the asset had none, and it understates the risk of that asset over those rows. `mu` and `sigma` stay the estimates that the recursion made from the rows it saw. The fill is silent when the filled share of the asset's observations is at or below the `fill_limit` field of the prior, warns above it, and raises for any fill under `strict`. `fill_limit` defaults to `nothing`, and this family has no `CoveragePolicy` to derive a limit from, so every fill warns.

# Mathematical definition

A valid observation of asset ``i`` has a finite return, and the active mask marks the asset active there. At each valid observation the state of asset ``i`` takes the step

```math
\\begin{align}
S_i &\\leftarrow \\lambda S_i + (1 - \\lambda) r_{i, t}\\,, \\\\
n_i &\\leftarrow n_i + 1\\,.
\\end{align}
```

The state starts at ``S_i = 0`` and ``n_i = 0``. After ``n_i`` valid observations it is

```math
\\begin{align}
S_i &= (1 - \\lambda) \\sum_{k=0}^{n_i - 1} \\lambda^{k} r_{i, t_{n_i - k}}\\,, \\\\
\\hat{\\mu}_i &= \\frac{S_i}{1 - \\lambda^{n_i}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_ew])
  - ``S_i``: State of asset ``i``, the exponentially weighted sum of its valid returns.
  - $(math_dict[:n_i_ew])
  - ``r_{i, t}``: Return of asset ``i`` at observation ``t``.
  - ``t_k``: Observation of the ``k``-th valid return of asset ``i``.
  - ``\\hat{\\mu}_i``: Expected return of asset ``i``.

The weights of ``S_i`` sum to ``1 - \\lambda^{n_i}`` and not to one, because the state starts at zero. The division by ``1 - \\lambda^{n_i}`` makes them sum to one, so ``\\hat{\\mu}_i`` is a weighted mean of the valid returns with weights proportional to ``\\lambda^{k}``. When every asset has the same ``n``, this is the ordinary adjusted exponentially weighted mean. The correction is the first power of ``1 - \\lambda^{n_i}``. A second-moment estimator corrects by a different power, so the two forms are not interchangeable.

An observation that is not valid changes the state in one of two ways:

  - An active asset with a return that is not finite is on a holiday. Its ``S_i`` and ``n_i`` do not change.
  - An asset that the active mask marks inactive, after it was active at the observation before, is reset to ``S_i = 0`` and ``n_i = 0``. If it becomes active again, the correction starts again.

``\\hat{\\mu}_i`` is `NaN` when ``n_i`` is less than `min_obs`, or when asset ``i`` is inactive at the last observation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpWeightedExpectedReturns(;
        decay::Number = exp2(-inv(40.0)),
        min_obs::Integer = round(Int, max(1, inv(log2(inv(decay))))),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> ExpWeightedExpectedReturns

Keywords correspond to the struct's fields.

The default `min_obs` is the half-life ``h = 1 / \\log_2(1 / \\lambda)`` rounded to the nearest integer, and at least one. The half-life is the count of observations over which a weight falls to one half, so after ``h`` valid observations the weight ``\\lambda^{n_i}`` that the cold start removes is one half. The default `decay` is ``2^{-1/40}``, a half-life of 40. The rule rounds, and does not truncate, because ``1 / \\log_2(1 / \\lambda)`` of ``\\lambda = 2^{-1/h}`` often comes back a small amount below the integer ``h``.

## Validation

  - $(val_dict[:decay])
  - $(val_dict[:min_obs])

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
        assert_unit_interval(decay, :decay)
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

Holds the running state of an [`ExpWeightedExpectedReturns`](@ref) fit, the ``S_i``, ``n_i`` and active flag of each asset.

The struct is immutable, and each field is an array that a fold changes in place. [`partial_fit!`](@ref) keeps the state in the `cache` field of the estimator, and `mean(me, state)` reads the expected returns out of it. [`ExpWeightedExpectedReturns`](@ref) states the mathematics.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ExpWeightedExpectedReturns`](@ref)
  - [`exp_weighted_pass!`](@ref)
  - [`exp_weighted_moment`](@ref)
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

Folds one observation into the state of an exponentially weighted mean.

[`ExpWeightedExpectedReturns`](@ref) states the step and the rules for an observation that is not valid.

# Algorithm

 1. Mark each asset valid when its return is finite and `active_mask` marks it active. With `active_mask = nothing`, every asset is active.
 2. If `active_mask` is not `nothing`, find each asset that `cache.active` marks active and `active_mask` marks inactive. Set its `mu` and its `obs_count` to zero. Then copy `active_mask` into `cache.active`. With `active_mask = nothing`, set every entry of `cache.active` to `true`.
 3. If no asset is valid, return `cache`.
 4. For each valid asset, set `mu = decay * mu + (1 - decay) * x`, and add one to its `obs_count`. Every other asset keeps its `mu` and its `obs_count`.

# Arguments

  - `cache::ExpWeightedExpectedReturnsState`: State to fold the observation into. The method changes its arrays in place.
  - `me::ExpWeightedExpectedReturns`: Exponentially weighted expected returns estimator.
  - `X::VecNum`: Returns of the assets at one observation.
  - `active_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask of the assets that are active at this observation. With `nothing` every asset is active, so a return that is not finite is a holiday.

# Returns

  - `cache::ExpWeightedExpectedReturnsState`: The same state, after the observation.

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

    cache.mu[valid] .= me.decay .* view(cache.mu, valid) .+
                       (one(me.decay) - me.decay) .* view(X, valid)
    cache.obs_count[valid] .+= 1

    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run one forward pass of an exponentially weighted update over the observations of `X`, and call `f` after each observation.

Each verb of [`ExpWeightedExpectedReturns`](@ref) runs the recursion through this method. `f` receives the index of the observation and the cache after that observation, so a caller can read the mean at every observation in one pass.

# Algorithm

 1. Check `dims`, and check that `active_mask` has the size of `X`.
 2. Count the assets `N` along the dimension that is not `dims`.
 3. If `state` is `nothing`, make a cold state: `mu` is `N` zeros, `obs_count` is `N` zeros, and `active` is `N` values `true`. The element type of `mu` is [`float_if_integer`](@ref) of `eltype(X)`, so an integer `X` gives a float state, a `Float32` `X` gives a `Float32` state, and a `Rational` `X` gives an exact state. Otherwise, check that `state` holds `N` assets, and continue from it.
 4. For each observation `i` along `dims`, in order, fold the observation and its row of `active_mask` into the cache with [`process_observation!`](@ref). Then call `f(i, cache)`.
 5. Return the cache.

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
        # A mean holds a fraction, so an integer sample takes a float state, and every
        # other sample keeps its own type: a `Float32` sample stays `Float32`.
        ExpWeightedExpectedReturnsState(zeros(float_if_integer(eltype(X)), N),
                                        zeros(Int, N), trues(N))
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

It calls the method that takes a callback, with a callback that does nothing. A verb that needs only the last cache calls this method.

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

The method reads the cache and does not write it, so the same cache answers this call after every observation of a forward pass. [`ExpWeightedExpectedReturns`](@ref) states the mathematics.

# Algorithm

 1. Copy `cache.mu`.
 2. For each asset with `obs_count > 0`, divide its entry by `1 - decay^obs_count`. An asset with `obs_count = 0` keeps its zero.
 3. Set to `NaN` each asset that `cache.active` marks inactive, or that has `obs_count < est.min_obs`.
 4. Return the vector.

# Arguments

  - `cache::ExpWeightedExpectedReturnsState`: State to read.
  - `est::ExpWeightedExpectedReturns`: Exponentially weighted expected returns estimator. The method reads its `decay` and its `min_obs`.

# Returns

  - `mu::Vector{<:Number}`: Per-asset expected returns vector. An asset that is inactive, or that has fewer than `est.min_obs` valid observations, is `NaN`.

# Related

  - [`ExpWeightedExpectedReturnsState`](@ref)
  - [`ExpWeightedExpectedReturns`](@ref)
  - [`exp_weighted_pass!`](@ref)
"""
function exp_weighted_moment(cache::ExpWeightedExpectedReturnsState,
                             est::ExpWeightedExpectedReturns)
    mu = copy(cache.mu)
    counted = cache.obs_count .> zero(eltype(cache.obs_count))
    # `0 < decay < 1`, so `1 - decay^n >= 1 - decay > 0` in floating point as well.
    mu[counted] ./= one(est.decay) .- est.decay .^ view(cache.obs_count, counted)
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

[`ExpWeightedExpectedReturns`](@ref) states the mathematics.

# Algorithm

 1. Fold every observation of `X` into a cold state with [`exp_weighted_pass!`](@ref).
 2. Read the mean out of the last state with [`exp_weighted_moment`](@ref).

# Arguments

  - `me`: Exponentially weighted expected returns estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`. An asset whose entry is `false` is inactive at that observation. Its state is reset, and its answer is `NaN` while it stays inactive. With `nothing` every asset is active, so a return that is not finite is a holiday and the mean does not change.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `mu::Vector{<:Number}`: Per-asset expected returns vector of length `assets`. An asset with fewer than `me.min_obs` valid observations is `NaN`, and so is an asset that is inactive at the last observation.

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

The estimator reads the active mask of the panel itself. So this method replaces the generic method, which reduces the sample to the Coverage Universe and expands the answer back. The answer covers every asset of the panel. A young asset that lists inside the window gets an answer from the observations it has, and it is `NaN` only while it has fewer than `me.min_obs` valid observations.

# Arguments

  - `me`: Exponentially weighted expected returns estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `mu::Vector{<:Number}`: Per-asset expected returns vector of length `assets`. An asset with fewer than `me.min_obs` valid observations is `NaN`, and so is an asset that is inactive at the last observation.

# Related

  - [`ExpWeightedExpectedReturns`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`Statistics.mean(me::AbstractExpectedReturnsEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.mean(me::ExpWeightedExpectedReturns, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
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

The estimator keeps the state in its `cache` field, so a second call continues the recursion and does not start it again. A partial-fit state is the one Result that an estimator can hold.

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

The entries of `x` are the returns of the assets at one observation. The method folds `x` as one row of the matrix method, for a caller that gets the observations one at a time.

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

The method calls [`exp_weighted_moment`](@ref), and [`ExpWeightedExpectedReturns`](@ref) states the mathematics.

# Arguments

  - `me`: Exponentially weighted expected returns estimator.
  - `state::ExpWeightedExpectedReturnsState`: The state to read.
  - $(arg_dict[:ignkwargs])

# Returns

  - `mu::Vector{<:Number}`: Per-asset expected returns vector. An asset that is inactive, or that has fewer than `me.min_obs` valid observations, is `NaN`.

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

[`partial_fit!`](@ref) leaves the state in the `cache` field, and this method reads the expected returns out of it. An estimator that has had no observation holds no state, so the method refuses the call and does not answer with a zero.

# Arguments

  - `me`: Exponentially weighted expected returns estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `me.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `mu::Vector{<:Number}`: Per-asset expected returns vector of length `assets`. An asset that is inactive, or that has fewer than `me.min_obs` valid observations, is `NaN`.

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

The state of two consecutive blocks is ``S = \\lambda^{n_b} S_a + S_b``, where ``S_a`` is the state of the first block, ``S_b`` the state of the second block from a cold start, and ``n_b`` the count of valid observations of the asset in the second block. That holds only when no asset resets inside the second block. The state records the count after a reset and not the reset itself, so the two cases cannot be told apart, and a merge would keep a history that the reset removed. Fold the second block into the first with [`partial_fit!`](@ref) instead.

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

This is the `copy` method of the [`AbstractPartialFitState`](@ref) interface. [`partial_fit`](@ref) calls it before it folds, so the state of the caller does not change. The method copies each of the three arrays.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::ExpWeightedExpectedReturnsState`: A new state, equal to `x`, with new arrays.

# Related

  - [`ExpWeightedExpectedReturnsState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::ExpWeightedExpectedReturnsState)
    return ExpWeightedExpectedReturnsState(copy(x.mu), copy(x.obs_count), copy(x.active))
end

# An exponentially weighted recursion folds in every configuration (see
# [`supports_partial_fit`](@ref)).
function supports_partial_fit(::ExpWeightedExpectedReturns)
    return true
end
export ExpWeightedExpectedReturns
