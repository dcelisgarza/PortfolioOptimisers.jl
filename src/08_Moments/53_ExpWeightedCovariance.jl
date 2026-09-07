"""
$(DocStringExtensions.TYPEDEF)

Estimates a covariance matrix by an exponentially weighted recursion that freezes on a holiday and resets on an inactive period.

Each observation updates the sub-block of the assets that are valid at it, so a gap never reaches an entry it did not touch. The recursion is seeded at zero, and the output divides out the damping that the cold start costs through a congruence transform, which keeps the state positive semidefinite and leaves every correlation unchanged.

Keeping a young asset investable has a cost the prior pays for it. A prior fitted with this estimator zero-fills the rows the asset was missing through [`scenario_fill`](@ref), because every consumer of a Prior Result reads its returns matrix; a scenario-based measure then reads a zero return where the asset had none and understates that asset's risk over those rows, while the covariance stays the estimate this recursion made from the rows it saw. The fill is silent at or below [`SCENARIO_FILL_LIMIT`](@ref), warns above it, and refuses any fill under `strict`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpWeightedCovariance(;
        decay::Number = exp2(-inv(40.0)),
        min_obs::Integer = round(Int, max(1, inv(log2(inv(decay))))),
        centred::Bool = false,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> ExpWeightedCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:decay])
  - `min_obs > 0`.

# Mathematical definition

The internal state after the valid observations of each asset is

```math
S = (1 - \\lambda) \\sum_{k} \\lambda^{k} e_{k} e_{k}^{\\top},
```

taken on the sub-block of the assets that are valid at each observation, and the reported covariance is the congruence transform

```math
\\hat{\\Sigma} = D S D, \\qquad D = \\operatorname{diag}\\left(\\frac{1}{\\sqrt{1 - \\lambda^{n_i}}}\\right).
```

Where:

  - ``\\lambda``: `decay`.
  - ``e_{k}``: the returns of the valid assets at the observation ``k``, less the running location where `centred` is `false`.
  - ``n_i``: the count of valid observations of asset ``i``.

A congruence transform preserves positive semidefiniteness, so ``\\hat{\\Sigma}`` inherits the property from ``S``, and it cancels in a correlation, so the correction moves no correlation. The correction is the **square root** of ``1 - \\lambda^{n_i}``, where a first-moment estimator uses the first power.

# Examples

```jldoctest
julia> ce = ExpWeightedCovariance();

julia> ce.decay ≈ exp2(-inv(40.0))
true

julia> ce.min_obs
40
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`ExpWeightedCovarianceState`](@ref)
  - [`ExpWeightedExpectedReturns`](@ref)
  - [`ExpWeightedVariance`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`partial_fit!`](@ref)
  - [`scenario_fill`](@ref)
  - [`SCENARIO_FILL_LIMIT`](@ref)
"""
@concrete struct ExpWeightedCovariance <: AbstractCovarianceEstimator
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
    function ExpWeightedCovariance(decay::Number, min_obs::Integer, centred::Bool,
                                   cache::Option{<:AbstractPartialFitState})
        assert_nonempty_gt0_finite_val(decay, :decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        return new{typeof(decay), typeof(min_obs), typeof(centred), typeof(cache)}(decay,
                                                                                   min_obs,
                                                                                   centred,
                                                                                   cache)
    end
end
function ExpWeightedCovariance(; decay::Number = exp2(-inv(40.0)),
                               min_obs::Integer = round(Int, max(1, inv(log2(inv(decay))))),
                               centred::Bool = false,
                               cache::Option{<:AbstractPartialFitState} = nothing)::ExpWeightedCovariance
    return ExpWeightedCovariance(decay, min_obs, centred, cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Internal mutable cache for the online covariance update in [`ExpWeightedCovariance`](@ref).

This type is an implementation detail and is not intended for direct use.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ExpWeightedCovariance`](@ref)
"""
@concrete struct ExpWeightedCovarianceState <: AbstractPartialFitState
    """
    $(field_dict[:ra_covariance])
    """
    covariance
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

Processes a single observation row (or column) to update the online covariance cache.

An asset is valid when its return is finite and the active mask admits it, and the update touches the sub-block of the valid assets alone. An active asset with a non-finite return freezes every entry that touches it, and an asset that has just become inactive has its whole row and column reset to the cold state so that the bias correction restarts if it lists again.

# Arguments

  - `cache::ExpWeightedCovarianceState`: Online covariance computation cache (mutated).
  - `ce::ExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Returns vector for the current observation.
  - `active_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask of currently active assets. With `nothing` every asset is active, so a non-finite return reads as a holiday.

# Returns

  - `cache::ExpWeightedCovarianceState`: The cache to read on and to pass to the next observation. Every field is an array that is mutated in place.

# Related

  - [`ExpWeightedCovarianceState`](@ref)
  - [`ExpWeightedCovariance`](@ref)
"""
function process_observation!(cache::ExpWeightedCovarianceState, ce::ExpWeightedCovariance,
                              X::VecNum, active_mask::Option{<:AbstractVector{<:Bool}})
    finite_mask = isfinite.(X)
    valid = isnothing(active_mask) ? finite_mask : (finite_mask .& active_mask)

    if !isnothing(active_mask)
        newly_inactive = .!active_mask .& cache.active
        if any(newly_inactive)
            cache.covariance[newly_inactive, :] .= zero(eltype(cache.covariance))
            cache.covariance[:, newly_inactive] .= zero(eltype(cache.covariance))
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

    idx = findall(valid)
    e = view(Xi, idx)
    block = view(cache.covariance, idx, idx)
    block .= ce.decay * block + (one(ce.decay) - ce.decay) * (e * transpose(e))
    cache.obs_count[idx] .+= 1

    return cache
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Covariance method of [`exp_weighted_pass!`](@ref). Runs one forward pass of the online covariance update over the observations of `X`, and calls `f` after each observation.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`ExpWeightedCovarianceState`](@ref)
  - [`exp_weighted_pass!`](@ref)
"""
function exp_weighted_pass!(f, est::ExpWeightedCovariance, X::MatNum, dims::Int,
                            active_mask::Option{<:AbstractMatrix{<:Bool}},
                            state::Option{<:ExpWeightedCovarianceState} = nothing)
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
        ExpWeightedCovarianceState(zeros(eltype(X), N, N),
                                   est.centred ? zeros(eltype(X), N) : fill(NaN, N),
                                   zeros(Int, N), trues(N))
    else
        @argcheck(size(state.covariance, 1) == N,
                  DimensionMismatch("the state holds $(size(state.covariance, 1)) assets, and `X` holds $N"))
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

Covariance method of [`exp_weighted_pass!`](@ref). Runs one forward pass of the online covariance update over the observations of `X`, and reads no intermediate cache.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`ExpWeightedCovarianceState`](@ref)
  - [`exp_weighted_pass!`](@ref)
"""
function exp_weighted_pass!(est::ExpWeightedCovariance, X::MatNum, dims::Int,
                            active_mask::Option{<:AbstractMatrix{<:Bool}},
                            state::Option{<:ExpWeightedCovarianceState} = nothing)
    return exp_weighted_pass!((args...) -> nothing, est, X, dims, active_mask, state)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Covariance method of [`exp_weighted_moment`](@ref). Reads the exponentially weighted covariance out of a cache, as it stands.

Applies the congruence correction, blanks the whole row and column of every asset that is not ready, and re-symmetrises the block that is left, so no transpose can pull a `NaN` into it. The cache is read, never written, so the same cache answers this call after every observation of a forward pass.

# Returns

  - `sigma::MatNum`: Covariance matrix. The row and the column of an asset that is inactive, or that carries fewer than `est.min_obs` valid observations, are `NaN`.

# Related

  - [`ExpWeightedCovarianceState`](@ref)
  - [`ExpWeightedCovariance`](@ref)
  - [`exp_weighted_moment`](@ref)
"""
function exp_weighted_moment(cache::ExpWeightedCovarianceState, est::ExpWeightedCovariance)
    T = eltype(cache.covariance)
    sigma = copy(cache.covariance)
    counted = cache.obs_count .> zero(eltype(cache.obs_count))
    correction = ones(T, length(counted))
    correction[counted] .= inv.(sqrt.(max.(one(est.decay) .-
                                           est.decay .^ view(cache.obs_count, counted),
                                           eps(T))))
    sigma .*= correction .* transpose(correction)
    not_ready = .!cache.active .| (cache.obs_count .< est.min_obs)
    if any(not_ready)
        sigma[not_ready, :] .= T(NaN)
        sigma[:, not_ready] .= T(NaN)
    end
    ready = .!not_ready
    if any(ready)
        idx = findall(ready)
        block = sigma[idx, idx]
        sigma[idx, idx] = (block + transpose(block)) / 2
    end

    return sigma
end
"""
    Statistics.cov(
        ce::ExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> MatNum

Compute the exponentially weighted covariance matrix.

Iterates over the observation dimension of `X`, updating an online covariance cache at each step, then applies the congruence correction and blanks every asset that is not ready.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`. An asset whose entry is `false` is inactive at that observation: its row and column are reset and its answer is `NaN` while it stays inactive. With `nothing` every asset is active, so a non-finite return reads as a holiday and every entry that touches the asset freezes.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `sigma::MatNum`: Covariance matrix of size `assets × assets`. The row and the column of an asset with fewer than `ce.min_obs` valid observations are `NaN`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = ExpWeightedCovariance(; decay = 0.9, min_obs = 2);

julia> size(cov(ce, X))
(2, 2)
```

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`ExpWeightedCovarianceState`](@ref)
  - [`Statistics.cor(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = exp_weighted_pass!(ce, X, dims, active_mask)
    if !ce.centred && any(.!cache.active)
        cache.location[.!cache.active] .= NaN
    end

    return exp_weighted_moment(cache, ce)
end
"""
    Statistics.cor(
        ce::ExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> MatNum

Compute the exponentially weighted correlation matrix.

This is the covariance of the same call, rescaled to a unit diagonal. The congruence correction cancels in the rescale, so the correlation is the correlation of the raw state.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`.
  - $(arg_dict[:ignkwargs])

# Returns

  - `rho::MatNum`: Correlation matrix of size `assets × assets`. The row and the column of an asset that is not ready are `NaN`, and so is its diagonal entry.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`regime_adjusted_correlation`](@ref): the shared rescale, which keeps a blanked asset `NaN` on the diagonal.
"""
function Statistics.cor(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    return regime_adjusted_correlation(cov(ce, X; dims = dims, active_mask = active_mask,
                                           kwargs...))
end
"""
    variance_series(
        ce::ExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Matrix{<:Number}

Compute the point-in-time exponentially weighted variance series.

Row `t` holds the diagonal of what `cov` returns for the first `t` observations of `X`, so no row reads an observation after its own. The update is a recursion over one observation, so this method overrides the expanding-window fallback with a **single forward pass**: it reads the cache after each observation instead of refitting.

The fallback cannot answer this estimator. It slices `X` once per row and passes every keyword unsliced, so a mask of the whole window meets a window of `t` observations and the size check refuses the call.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
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

  - [`ExpWeightedCovariance`](@ref)
  - [`exp_weighted_moment(cache::ExpWeightedCovarianceState, est::ExpWeightedCovariance)`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_dims(dims)
    val = Matrix{eltype(X)}(undef, size(X, dims), size(X, setdiff((1, 2), (dims,))[1]))
    exp_weighted_pass!(ce, X, dims, active_mask) do i, cache
        val[i, :] = LinearAlgebra.diag(exp_weighted_moment(cache, ce))
        return nothing
    end

    return isone(dims) ? val : permutedims(val)
end
"""
    Statistics.cov(
        ce::ExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the exponentially weighted covariance from a window of an Asset Panel.

This estimator is mask-aware, so it overrides the reduce-and-expand root of the verb and reads the panel's active mask itself. The answer therefore lives on the whole universe rather than on the Coverage Universe: a young asset that lists inside the window is answered from the observations it has, and it is `NaN` only while it stays below `ce.min_obs`.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `sigma::MatNum`: Covariance matrix of size `assets × assets`.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    amsk, _ = panel_moment_masks(pnl)
    return Statistics.cov(ce, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    Statistics.cor(
        ce::ExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the exponentially weighted correlation from a window of an Asset Panel.

This is the covariance of the same call, rescaled to a unit diagonal, and it reads the panel's active mask through the same override.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `rho::MatNum`: Correlation matrix of size `assets × assets`.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    amsk, _ = panel_moment_masks(pnl)
    return Statistics.cor(ce, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    Statistics.var(
        ce::ExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the marginal exponentially weighted variance from a window of an Asset Panel.

This is the diagonal of the covariance of the same call, and it reads the panel's active mask through the same override.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `var::MatNum`: Marginal variance, as a row where `dims` is `1` and as a column otherwise.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.var(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    amsk, _ = panel_moment_masks(pnl)
    return Statistics.var(ce, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    Statistics.std(
        ce::ExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the marginal exponentially weighted volatility from a window of an Asset Panel.

This is the square root of the diagonal of the covariance of the same call, and it reads the panel's active mask through the same override.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `std::MatNum`: Marginal volatility, as a row where `dims` is `1` and as a column otherwise.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`Statistics.var(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.std(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    amsk, _ = panel_moment_masks(pnl)
    return Statistics.std(ce, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    variance_series(
        ce::ExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> Matrix{<:Number}

Compute the point-in-time exponentially weighted variance series from a window of an Asset Panel.

This is the diagonal of the covariance of the same call, read after each observation, and it reads the panel's active mask through the same override. A Descriptor that holds this estimator therefore keeps a number for a young asset, where the reduce-and-expand root would reduce every window and answer `NaN`.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `val::Matrix{<:Number}`: Variance series on the full asset universe, shaped as `(T, N)` if
    `dims == 1` or `(N, T)` if `dims == 2`.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`variance_series(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                         dims::Int = 1, kwargs...)
    amsk, _ = panel_moment_masks(pnl)
    return variance_series(ce, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    partial_fit!(
        ce::ExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> ExpWeightedCovariance

Fold a block of observations into the estimator's own online covariance state.

The estimator carries the state in its `cache` field, so a second call continues the recursion rather than restarting it. This is the exception ADR 0106 states: a partial-fit state is the one Result an estimator may hold.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `X`.

# Returns

  - `ce::ExpWeightedCovariance`: The estimator, with `cache` holding the state after the block.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`ExpWeightedCovarianceState`](@ref)
  - [`Statistics.cov(ce::ExpWeightedCovariance; kwargs...)`](@ref)
"""
function partial_fit!(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = exp_weighted_pass!(ce, X, dims, active_mask, ce.cache)
    Accessors.@reset ce.cache = cache
    return ce
end
"""
    partial_fit!(
        ce::ExpWeightedCovariance,
        x::VecNum;
        active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
        kwargs...
    ) -> ExpWeightedCovariance

Fold one observation into the estimator's own online covariance state.

The entries of `x` are the assets of a single observation, which is the row the matrix method folds one at a time. This is the shape a caller has when the observations arrive one by one.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - `x::VecNum`: One observation, with one entry per asset.
  - `active_mask`: Optional boolean vector with the same length as `x`.
  - $(arg_dict[:ignkwargs])

# Validation

  - If `active_mask` is not `nothing`, `length(x) == length(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `x`.

# Returns

  - `ce::ExpWeightedCovariance`: The estimator, with `cache` holding the state after the observation.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = ExpWeightedCovariance(; decay = 0.9, min_obs = 2);

julia> one_at_a_time = foldl((c, i) -> partial_fit!(c, view(X, i, :)), axes(X, 1); init = ce);

julia> isequal(cov(one_at_a_time), cov(ce, X))
true
```

# Related

  - [`partial_fit!(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`ExpWeightedCovariance`](@ref)
"""
function partial_fit!(ce::ExpWeightedCovariance, x::VecNum;
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
    return partial_fit!(ce, permutedims(x); dims = 1,
                        active_mask = if isnothing(active_mask)
                            nothing
                        else
                            permutedims(active_mask)
                        end)
end
"""
    Statistics.cov(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState; kwargs...) -> MatNum

Read the exponentially weighted covariance out of a state the caller holds.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - `state::ExpWeightedCovarianceState`: The state to read.
  - $(arg_dict[:ignkwargs])

# Returns

  - `sigma::MatNum`: Covariance matrix.

# Related

  - [`ExpWeightedCovarianceState`](@ref)
  - [`exp_weighted_moment`](@ref)
"""
function Statistics.cov(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState;
                        kwargs...)
    return exp_weighted_moment(state, ce)
end
"""
    Statistics.cov(ce::ExpWeightedCovariance; kwargs...) -> MatNum

Read the exponentially weighted covariance out of the estimator's own state.

The one-argument form is what an incremental fit answers: [`partial_fit!`](@ref) leaves the state in the `cache` field, and this verb turns it into the ordinary answer. An estimator that has been given no observation carries no state, so the call is refused rather than answered with a zero.

# Arguments

  - `ce`: Exponentially weighted covariance estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `sigma::MatNum`: Covariance matrix of size `assets × assets`.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(ExpWeightedCovariance(; decay = 0.9, min_obs = 2), X);

julia> size(cov(ce))
(2, 2)

julia> cov(ExpWeightedCovariance())
ERROR: ArgumentError: `ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `cov(ce, X)` for a fit over a whole sample.
[...]
```

# Related

  - [`partial_fit!(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`Statistics.cov(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState; kwargs...)`](@ref)
"""
function Statistics.cov(ce::ExpWeightedCovariance; kwargs...)
    state = ce.cache
    @argcheck(!isnothing(state),
              ArgumentError("`ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `cov(ce, X)` for a fit over a whole sample."))
    return cov(ce, state)
end
"""
    Statistics.cor(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState; kwargs...) -> MatNum

Read the exponentially weighted correlation out of a state the caller holds.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - `state::ExpWeightedCovarianceState`: The state to read.
  - $(arg_dict[:ignkwargs])

# Returns

  - `rho::MatNum`: Correlation matrix.

# Related

  - [`ExpWeightedCovarianceState`](@ref)
  - [`Statistics.cov(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState; kwargs...)`](@ref)
"""
function Statistics.cor(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState;
                        kwargs...)
    return regime_adjusted_correlation(cov(ce, state; kwargs...))
end
"""
    Statistics.cor(ce::ExpWeightedCovariance; kwargs...) -> MatNum

Read the exponentially weighted correlation out of the estimator's own state.

# Arguments

  - `ce`: Exponentially weighted covariance estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `rho::MatNum`: Correlation matrix of size `assets × assets`.

# Related

  - [`Statistics.cov(ce::ExpWeightedCovariance; kwargs...)`](@ref)
  - [`ExpWeightedCovarianceState`](@ref)
"""
function Statistics.cor(ce::ExpWeightedCovariance; kwargs...)
    return regime_adjusted_correlation(cov(ce; kwargs...))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`ExpWeightedCovarianceState`](@ref).

An exponentially weighted state folds forward exactly, `S = λ^{n_b} S_a + S_b`, but only while no asset resets inside the second block. The state records the count that a reset zeroed and not the reset itself, so the two cases are indistinguishable after the fact and a merge would silently keep a history the reset discarded. Fold the second block into the first with `partial_fit!` instead.

# Arguments

  - `a`: State of the first block.
  - `b`: State of the second block.

# Validation

  - The pair is refused with an `ArgumentError`.

# Related

  - [`ExpWeightedCovarianceState`](@ref)
  - [`merge_states`](@ref)
  - [`assert_mergeable_states`](@ref)
"""
function merge_states(a::ExpWeightedCovarianceState, b::ExpWeightedCovarianceState)
    assert_mergeable_states(a, b)
    return throw(ArgumentError("an `ExpWeightedCovarianceState` pair does not merge, because the state does not record whether an asset reset inside the second block. A reset zeroes the count that the fold would read, so the two histories are indistinguishable. Fold the second block into the first with `partial_fit!` instead."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`ExpWeightedCovarianceState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. Every field is an array, and every one is copied.

# Arguments

  - `x`: The cache to copy.

# Returns

  - `state::ExpWeightedCovarianceState`: A fresh cache, equal to `x`, whose arrays are fresh.

# Related

  - [`ExpWeightedCovarianceState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::ExpWeightedCovarianceState)
    return ExpWeightedCovarianceState(copy(x.covariance), copy(x.location),
                                      copy(x.obs_count), copy(x.active))
end

export ExpWeightedCovariance
