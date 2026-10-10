"""
$(DocStringExtensions.TYPEDEF)

Estimates per-asset variance by an exponentially weighted recursion that freezes on a holiday and resets on an inactive period.

The recursion starts at zero, so a newly listed asset starts from a cold state, and the output divides out the weight that the cold start lacks. An asset with fewer than `min_obs` valid observations since its last reset is `NaN`, and so is an asset that the active mask leaves inactive at the last observation.

A young asset stays investable, and the returns matrix of the prior carries the cost. A prior fitted with this estimator fills the rows that the asset was missing with zero through [`scenario_fill`](@ref), because every consumer of a Prior Result reads its returns matrix. A scenario-based measure then reads a zero return where the asset had none, so it understates the risk of that asset over those rows. The variance stays the estimate that this recursion made from the rows it saw. The fill is silent up to the `fill_limit` field of the fitting prior, a share of the asset's own observations. Above that share it warns, and `strict` refuses any fill. `fill_limit` defaults to `nothing`, and this family carries no `CoveragePolicy` to derive a limit from, so every fill warns.

The estimator has no window of its own. [`WindowedVariance`](@ref) is its window: the wrapper keeps the last `window` observations of the returns and of the active mask, and the recursion starts cold at the first of them. In a variance series, row `t` keeps the window that ends at `t`. The example below puts the windowed estimator in the variance slot of a [`CrossSectionalFactorPrior`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpWeightedVariance(;
        decay::Number = exp2(-inv(40.0)),
        min_obs::Integer = round(Int, max(1, decay_half_life(decay))),
        centring::AbstractCentring = EstimatedCentring(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> ExpWeightedVariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:decay])
  - `min_obs > 0`.

# Mathematical definition

Let ``x_{i, k}`` be the ``k``-th valid return of asset ``i`` since its last reset, ``k = 1, \\ldots, n_i``. Under [`EstimatedCentring`](@ref), the default, the location is the exponentially weighted mean of the returns, divided by the sum of its weights. The deviation of a return is taken from the location of the returns before it, and its square is divided by its exact factor, because the location is itself an estimate,

```math
\\begin{align}
m_{i, k} &= \\frac{\\sum_{j = 1}^{k} \\lambda^{k - j} x_{i, j}}{\\sum_{j = 1}^{k} \\lambda^{k - j}}\\,, \\\\
e_{i, k}^{2} &= \\frac{\\left(x_{i, k} - m_{i, k - 1}\\right)^{2}}{1 + 1 / n^{\\mathrm{eff}}_{k - 1}}\\,, \\quad k = 2, \\ldots, n_i\\,.
\\end{align}
```

The first return gives no term, so the lag of the centring is ``\\ell = 1``. Under [`PreCentred`](@ref) the deviation is the return, ``e_{i, k} = x_{i, k}`` for ``k = 1, \\ldots, n_i``, and ``\\ell = 0``. The internal state of asset ``i`` is the exponentially weighted sum of its ``K_i = n_i - \\ell`` terms, and the reported variance divides out the weights that the cold start never accumulated,

```math
\\begin{align}
S_i &= (1 - \\lambda) \\sum_{k = \\ell + 1}^{n_i} \\lambda^{n_i - k} e_{i, k}^{2}\\,, \\\\
\\hat{\\sigma}^{2}_i &= \\frac{S_i}{1 - \\lambda^{K_i}}\\,.
\\end{align}
```

For returns that are independent in time, each term has the mean ``\\sigma^{2}_i``, so under `EstimatedCentring` the estimate is unbiased from the second return, whatever the mean. Under `PreCentred` it is too large by the square of the mean. [`ZeroStartCentring`](@ref) keeps the uncorrected rule: ``\\ell = 0``, the location takes the step ``m_{i, k} = \\lambda m_{i, k - 1} + (1 - \\lambda) x_{i, k}`` from ``m_{i, 0} = 0``, and ``e_{i, k} = x_{i, k} - m_{i, k - 1}`` with no factor.

Where:

  - $(math_dict[:lambda_ew])
  - $(math_dict[:n_i_ew])
  - ``x_{i, k}``: The ``k``-th valid return of asset ``i`` since its last reset.
  - ``m_{i, k}``: The location of asset ``i`` after ``k`` valid returns.
  - ``n^{\\mathrm{eff}}_{k}``: The Kish count of the weights of a location of ``k`` returns, so ``1 + 1 / n^{\\mathrm{eff}}_{k}`` is the factor of [`centring_factor`](@ref).
  - ``e_{i, k}^{2}``: The term of the ``k``-th valid return of asset ``i``.
  - ``\\ell``: The lag of the centring, the count of the leading returns that give no term.
  - ``K_i``: The count of the terms of asset ``i``.
  - ``S_i``: The internal state of asset ``i``, the `variance` field of [`ExpWeightedVarianceState`](@ref).
  - ``\\hat{\\sigma}^{2}_i``: The reported variance of asset ``i``.

# Examples

```jldoctest
julia> ce = ExpWeightedVariance();

julia> ce.decay ≈ exp2(-inv(40.0))
true

julia> ce.min_obs
40

julia> ve = WindowedVariance(; ve = ce, window = 500);

julia> pe = CrossSectionalFactorPrior(; factors = [\"market\" => ConstantExposure()], ve = ve);

julia> pe.ve.window
500
```

# Related

  - [`WindowedVariance`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`ExpWeightedVarianceState`](@ref)
  - [`ExpWeightedExpectedReturns`](@ref)
  - [`ExpWeightedCovariance`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`partial_fit!`](@ref)
  - [`scenario_fill`](@ref)
  - [`EmpiricalPrior`](@ref)
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
    $(field_dict[:centring])
    """
    centring
    """
    $(field_dict[:ew_cache])
    """
    cache
    function ExpWeightedVariance(decay::Number, min_obs::Integer,
                                 centring::AbstractCentring,
                                 cache::Option{<:AbstractPartialFitState})
        assert_unit_interval(decay, :decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        return new{typeof(decay), typeof(min_obs), typeof(centring), typeof(cache)}(decay,
                                                                                    min_obs,
                                                                                    centring,
                                                                                    cache)
    end
end
function ExpWeightedVariance(; decay::Number = exp2(-inv(40.0)),
                             min_obs::Integer = round(Int, max(1, decay_half_life(decay))),
                             centring::AbstractCentring = EstimatedCentring(),
                             cache::Option{<:AbstractPartialFitState} = nothing)::ExpWeightedVariance
    return ExpWeightedVariance(decay, min_obs, centring, cache)
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

Folds one observation into the online variance cache.

An asset is valid when its return is finite and the active mask admits it. A valid asset takes one step of the recursion, and an active asset with a non-finite return freezes. An asset that has just become inactive goes back to the cold state, so the bias correction starts again if it lists again.

# Algorithm

 1. Mark each asset whose return is finite and that `active_mask` admits, giving `valid`. With `nothing`, `valid` is the finite mask alone.
 2. Where `active_mask` is not `nothing`, mark each asset that was active before this observation and is inactive now, giving `newly_inactive`. Set its `variance` and `obs_count` to zero, and its `location` to `NaN`.
 3. Store `active_mask` in `active`. With `nothing`, set every entry of `active` to `true`.
 4. If no entry of `valid` is `true`, return the cache unchanged.
 5. Take the deviations, and the mask `dvalid` of the assets that give a term, with [`centring_location!`](@ref). It also moves the `location` of each valid asset.
 6. Divide the square of each deviation by its factor, [`centring_factor`](@ref) of the count before the observation. Give the `variance` of each asset of `dvalid` one step of the recursion of ``S_i``, and add one to the `obs_count` of each valid asset.
 7. Return the cache.

An active asset whose return is not finite is not in `valid`, so steps 5 and 6 leave its state as it was: the state freezes over the holiday. Under [`EstimatedCentring`](@ref) the first valid return of an asset moves its location and its count, and gives no term.

# Arguments

  - `cache::ExpWeightedVarianceState`: The online variance cache, which this call mutates.
  - `ce::ExpWeightedVariance`: Variance estimator configuration.
  - `X::VecNum`: Returns vector for the current observation.
  - `active_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask of currently active assets. An asset that becomes inactive has its variance, its location and its count reset. With `nothing` every asset is active, so a non-finite return reads as a holiday.

# Returns

  - `cache::ExpWeightedVarianceState`: The cache to read, and to pass to the next observation. Every field is an array that this call mutates in place.

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
            cache.location[newly_inactive] .= convert(eltype(cache.location), NaN)
            cache.obs_count[newly_inactive] .= 0
        end
        cache.active .= active_mask
    else
        cache.active .= true
    end

    if !any(valid)
        return cache
    end

    dev, dvalid = centring_location!(ce.centring, cache.location, cache.obs_count, X, valid,
                                     ce.decay)
    e2 = view(dev, dvalid) .^ 2 ./
         centring_factor.(Ref(ce.centring), ce.decay, view(cache.obs_count, dvalid))
    cache.variance[dvalid] .= ce.decay * view(cache.variance, dvalid) +
                              (one(ce.decay) - ce.decay) * e2
    cache.obs_count[valid] .+= 1

    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Variance method of [`exp_weighted_pass!`](@ref). Runs one forward pass of the online variance update over the observations of `X`, and calls `f` after each observation. The pass continues from `state` when the caller gives one, and starts from the cold state otherwise.

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

    # The state takes the type of `X`, widened to a float only when it is an integer, so an
    # integer panel gets a floating-point state and a `Float32` panel keeps a `Float32` one.
    T = float_if_integer(eltype(X))
    cache = if isnothing(state)
        ExpWeightedVarianceState(zeros(T, N), fill(convert(T, NaN), N), zeros(Int, N),
                                 trues(N))
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

Variance method of [`exp_weighted_pass!`](@ref). Runs one forward pass of the online variance update over the observations of `X`, and reads no intermediate cache.

# Related

  - [`ExpWeightedVariance`](@ref)
  - [`ExpWeightedVarianceState`](@ref)
  - [`exp_weighted_pass!`](@ref)
"""
function exp_weighted_pass!(est::ExpWeightedVariance, X::MatNum, dims::Int,
                            active_mask::Option{<:AbstractMatrix{<:Bool}},
                            state::Option{<:ExpWeightedVarianceState} = nothing)
    return exp_weighted_pass!((args...) -> nothing, est, X, dims, active_mask, state)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Variance method of [`exp_weighted_moment`](@ref). Reads the exponentially weighted variance out of a cache, as it stands.

Applies the cold-start bias correction and sets every asset that is not ready to `NaN`. This method reads the cache and never writes it, so the same cache answers this call after every observation of a forward pass.

# Algorithm

 1. Copy `cache.variance` into `variance`.
 2. Take the count of the terms ``K_i`` of each asset from its `obs_count` with [`centring_terms`](@ref), and mark each asset with a positive count, giving `counted`.
 3. Give each counted asset the cold-start `correction` ``1 / (1 - \\lambda^{K_i})``. Every other asset keeps a `correction` of one.
 4. Multiply `variance` by `correction`.
 5. Mark each asset that is inactive, or that [`centring_ready`](@ref) does not find ready, giving `not_ready`, and set its entry of `variance` to `NaN`.
 6. Return `variance`.

Because ``0 < \\lambda < 1``, ``1 - \\lambda^{K_i} \\geq 1 - \\lambda > 0`` for every ``K_i \\geq 1``, so the division needs no floor.

# Returns

  - `variance::Vector{<:Number}`: Per-asset variance vector. An asset that is inactive, or that carries fewer than `est.min_obs` valid observations, is `NaN`.

# Related

  - [`ExpWeightedVarianceState`](@ref)
  - [`ExpWeightedVariance`](@ref)
  - [`exp_weighted_moment`](@ref)
"""
function exp_weighted_moment(cache::ExpWeightedVarianceState, est::ExpWeightedVariance)
    variance = copy(cache.variance)
    K = centring_terms.(Ref(est.centring), cache.obs_count)
    counted = K .> zero(eltype(K))
    correction = ones(eltype(variance), length(variance))
    correction[counted] .= inv.(one(est.decay) .- est.decay .^ view(K, counted))
    variance .*= correction
    not_ready = .!cache.active .| .!centring_ready.(Ref(est.centring), cache.obs_count, est.min_obs)
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

Runs the recursion over the observations of `X` in one forward pass. Then it applies the cold-start bias correction and sets every asset that is not ready to `NaN`.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`. An asset whose entry is `false` is inactive at that observation. The estimator resets its state, and answers `NaN` for it while it stays inactive. With `nothing` every asset is active, so a non-finite return reads as a holiday and the variance freezes.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `var::Vector{<:Number}`: Per-asset variance vector of length `assets`. An asset with fewer than `ce.min_obs` valid observations since its last reset is `NaN`, and so is an asset that is inactive at the last observation.

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
    cache = exp_weighted_pass!(ce, X, dims, active_mask)
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
    variance_series(
        ce::ExpWeightedVariance,
        X::MatNum;
        dims::Int = 1,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Matrix{<:Number}

Compute the point-in-time exponentially weighted variance series.

Row `t` holds what `var` returns for the first `t` observations of `X`, so no row reads an observation after its own. The update is a recursion over one observation, so this method replaces the expanding-window fallback with one forward pass. It reads the cache after each observation instead of a refit.

The fallback cannot answer this estimator. It slices `X` once per row and passes every keyword unsliced, so a mask of the whole window meets a window of `t` observations and the size check refuses the call.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if
    `dims == 2`. An asset with fewer than `ce.min_obs` observations at row `t` is `NaN` there.

# Related

  - [`ExpWeightedVariance`](@ref)
  - [`exp_weighted_moment(cache::ExpWeightedVarianceState, est::ExpWeightedVariance)`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_dims(dims)
    T = float_if_integer(eltype(X))
    N = size(X, setdiff((1, 2), (dims,))[1])
    val = Matrix{T}(undef, size(X, dims), N)
    exp_weighted_pass!(ce, X, dims, active_mask) do i, cache
        val[i, :] = exp_weighted_moment(cache, ce)
        return nothing
    end

    return isone(dims) ? val : permutedims(val)
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

This estimator reads the active mask of the panel itself, so it overrides the reduce-and-expand root of the verb. Its answer covers the whole universe, not only the Coverage Universe. A young asset that lists inside the window gets an answer from the observations it has, and it is `NaN` only while it has fewer than `ce.min_obs`.

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
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
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
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.std(ce, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    variance_series(
        ce::ExpWeightedVariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> Matrix{<:Number}

Compute the point-in-time exponentially weighted variance series from a window of an Asset Panel.

This is the variance of the same call, read after each observation, and it reads the panel's active mask through the same override. A Descriptor that holds this estimator keeps a number for a young asset, where the reduce-and-expand root would reduce every window and answer `NaN`.

# Arguments

  - `ce`: Exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `val::Matrix{<:Number}`: Variance series on the full asset universe, shaped as `(T, N)` if
    `dims == 1` or `(N, T)` if `dims == 2`.

# Related

  - [`ExpWeightedVariance`](@ref)
  - [`variance_series(ce::ExpWeightedVariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::ExpWeightedVariance, X::MatNum, pnl::Option{<:AssetPanel};
                         dims::Int = 1, kwargs...)
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
    return variance_series(ce, X; dims = dims, active_mask = amsk, kwargs...)
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

The estimator carries the state in its `cache` field, so a second call continues the recursion rather than restarting it. A partial-fit state is the one Result an estimator may hold, and this is that exception.

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
function partial_fit!(ce::ExpWeightedVariance{<:Any, <:Any, <:Any,
                                              <:Option{<:ExpWeightedVarianceState}},
                      X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = exp_weighted_pass!(ce, X, dims, active_mask, ce.cache)
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
function partial_fit!(ce::ExpWeightedVariance{<:Any, <:Any, <:Any,
                                              <:Option{<:ExpWeightedVarianceState}},
                      x::VecNum; active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      kwargs...)
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

The one-argument form is what an incremental fit answers: [`partial_fit!`](@ref) leaves the state in the `cache` field, and this verb turns it into the ordinary answer. An estimator that has seen no observation carries no state, so this method refuses the call instead of answering zero.

# Arguments

  - `ce`: Exponentially weighted variance estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. Otherwise the method throws an `ArgumentError`.

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

  - `ce.cache` is not `nothing`. Otherwise the method throws an `ArgumentError`.

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

A state under [`PreCentred`](@ref) folds forward exactly, `S = λ^{n_b} S_a + S_b` with `n_b` the valid observations of the second block, but only while no asset resets inside that block. The state records the count that a reset zeroed and not the reset itself, so a merge cannot tell the two cases apart, and it would keep a history that the reset discarded. A state under [`EstimatedCentring`](@ref) does not fold at all, because each deviation of the second block reads a location that carries the first block. Fold the second block into the first with `partial_fit!` instead.

# Arguments

  - `a`: State of the first block.
  - `b`: State of the second block.

# Validation

  - The method throws an `ArgumentError` for every pair.

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

It is the `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. Every field is an array, and this method copies each one.

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

# An exponentially weighted recursion folds in every configuration (see
# [`supports_partial_fit`](@ref)).
function supports_partial_fit(::ExpWeightedVariance)
    return true
end
"""
    exp_weighted_variance_count(decay::Number, X::MatNum, centring::AbstractCentring)

Effective count and divisor of an exponentially weighted variance, for each column of `X`.

Each column enters the recursion on its finite rows alone, and the recursion divides out the weights that the cold start never accumulated. A column of ``n_{i}`` finite rows gives ``K_{i} = n_{i} - \\ell`` terms, with ``\\ell`` the lag of the centring of [`centring_lag`](@ref). The weights of the terms are proportional to ``\\lambda^{k}``, ``k = 0, \\ldots, K_{i} - 1``, and they sum to one. The count reads no active mask, so it does not start again at a reset. [`ExpWeightedVariance`](@ref) and [`RegimeAdjustedExpWeightedVariance`](@ref) share this count.

The Kish count of the weights of the terms is the count of the terms that the estimate is worth. Under [`PreCentred`](@ref) each term reads one observation, so it is both the effective count and the divisor. Under [`EstimatedCentring`](@ref) the location spends one observation, as the mean of a sample variance does, so the effective count is one more than the divisor. With equal weights the pair is ``(n_{i}, n_{i} - 1)``, the count and the divisor of a sample variance.

# Mathematical definition

```math
\\begin{align}
\\kappa_{i} &= \\dfrac{\\left(\\sum_{k=0}^{K_{i}-1} \\lambda^{k}\\right)^{2}}{\\sum_{k=0}^{K_{i}-1} \\lambda^{2k}} = \\dfrac{\\left(1 - \\lambda^{K_{i}}\\right)\\left(1 + \\lambda\\right)}{\\left(1 - \\lambda\\right)\\left(1 + \\lambda^{K_{i}}\\right)}\\,, \\\\
n_{i}^{\\mathrm{eff}} &= \\kappa_{i} + \\ell\\,, \\\\
m_{i} &= \\kappa_{i}\\,.
\\end{align}
```

Where:

  - ``\\kappa_{i}``: Kish count of the weights of the terms of asset ``i``.
  - ``n_{i}^{\\mathrm{eff}}``: Effective count of the observations of asset ``i``.
  - ``m_{i}``: The divisor of its variance.
  - ``n_{i}``: Number of finite rows of column ``i``.
  - ``\\ell``: The lag of the centring.
  - $(math_dict[:lambda_ew])

The count tends to ``(1 + \\lambda) / (1 - \\lambda)`` as ``n_{i}`` grows, which is about ``115`` at the default half-life of ``40`` observations.

# Arguments

  - `decay`: Decay of the recursion.
  - `X`: Data matrix `observations × assets`.
  - `centring`: The centring of the estimator.

# Returns

  - `(; n, m)::NamedTuple`: The effective count `n` and the divisor `m`, one entry per column of `X`.

# Related

  - [`variance_count`](@ref)
  - [`ExpWeightedVariance`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function exp_weighted_variance_count(decay::Number, X::MatNum, centring::AbstractCentring)
    m = map(axes(X, 2)) do i
        lk = decay^centring_terms(centring, count(isfinite, view(X, :, i)))
        return (one(lk) - lk) * (one(decay) + decay) /
               ((one(decay) - decay) * (one(lk) + lk))
    end
    return (; n = m .+ centring_lag(centring), m = m)
end
function variance_count(ve::ExpWeightedVariance, X::MatNum)
    return exp_weighted_variance_count(ve.decay, X, ve.centring)
end
export ExpWeightedVariance
