"""
$(DocStringExtensions.TYPEDEF)

Estimates a covariance matrix by an exponentially weighted recursion that freezes on a holiday and resets on an inactive period.

An entry of the state moves only at an observation where both of its assets are valid, so a gap in one asset leaves every entry of that asset unchanged. The state starts at zero. The estimate divides out the weight that the zero start lacks, and it divides the rows and the columns by the same factor, so every correlation stays unchanged.

A holiday can make the estimate indefinite. While one asset is on a holiday, the variance of another asset moves and their covariance does not, so the implied correlation can leave ``[-1, 1]``. Without a holiday the estimate is positive semidefinite. Wrap the estimator in an estimator that repairs the matrix, such as [`PortfolioOptimisersCovariance`](@ref), when a consumer needs a positive semidefinite matrix.

A young asset that stays investable has a cost for the prior. A prior fitted with this estimator fills the rows that the asset lacks with zeros through [`scenario_fill`](@ref), because every consumer of a Prior Result reads its returns matrix. A scenario-based measure then reads a zero return where the asset had none, and it understates the risk of that asset over those rows. The covariance stays the estimate that this recursion made from the rows it saw. The fill is silent at or below the `fill_limit` field of the fitting prior, which is a share of the observations of that asset. Above the limit it warns, and under `strict` it refuses every fill. `fill_limit` defaults to `nothing`, and this family has no `CoveragePolicy` to derive a limit from, so the prior names every fill.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExpWeightedCovariance(;
        decay::Number = exp2(-inv(40.0)),
        min_obs::Integer = round(Int, max(1, inv(log2(inv(decay))))),
        centred::Bool = false,
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> ExpWeightedCovariance

Keywords correspond to the struct's fields. The default `min_obs` is the half-life of `decay`, the count of observations after which a weight halves.

## Validation

  - $(val_dict[:decay])
  - $(val_dict[:min_obs])

# Mathematical definition

```math
\\begin{align}
m_{t,\\,i} &= (1 - \\lambda) \\sum_{u \\in \\mathcal{V}_i,\\, u < t} \\lambda^{c_i(u,\\, t)} x_{u,\\,i}\\,, \\\\
e_{t,\\,i} &= x_{t,\\,i} - m_{t,\\,i}\\,, \\\\
S_{ij} &= (1 - \\lambda) \\sum_{t \\in \\mathcal{V}_i \\cap \\mathcal{V}_j} \\lambda^{c_{ij}(t)} e_{t,\\,i} e_{t,\\,j}\\,, \\\\
\\hat{\\Sigma}_{ij} &= \\frac{S_{ij}}{\\sqrt{(1 - \\lambda^{n_i})(1 - \\lambda^{n_j})}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:lambda_ew])
  - $(math_dict[:x_ti_ret])
  - ``\\mathcal{V}_i``: Valid observations of asset ``i``. They are the observations after the last one at which the active mask excludes the asset, where the return of the asset is finite.
  - $(math_dict[:n_i_ew])
  - ``c_i(u,\\, t)``: Count of the observations of ``\\mathcal{V}_i`` that lie strictly between ``u`` and ``t``.
  - ``c_{ij}(t)``: Count of the observations of ``\\mathcal{V}_i \\cap \\mathcal{V}_j`` that lie after ``t``.
  - ``m_{t,\\,i}``: Running location of asset ``i`` before the observation ``t``. It is zero where `centred` is `true`.
  - ``e_{t,\\,i}``: Deviation of asset ``i`` at the observation ``t``.
  - ``S_{ij}``: Entry of the internal state.
  - ``\\hat{\\Sigma}_{ij}``: Entry of the estimate. It is `NaN` where asset ``i`` or asset ``j`` has fewer than `min_obs` valid observations, or where the active mask excludes it at the last observation.

The weights of ``S_{ii}`` sum to ``1 - \\lambda^{n_i}``, so ``\\hat{\\Sigma}_{ii}`` is a weighted mean of squared deviations whose weights sum to one. In matrix form ``\\hat{\\Sigma} = D S D`` with ``D = \\operatorname{diag}(1 / \\sqrt{1 - \\lambda^{n_i}})``. This congruence transform cancels in a correlation, and ``\\hat{\\Sigma}`` is positive semidefinite exactly when ``S`` is. ``S`` is positive semidefinite when each ``\\mathcal{V}_i`` is empty or an unbroken run of observations that ends at the last one, which holds when no asset has a holiday. Then every exponent ``c_{ij}(t)`` is the count of observations after ``t``, and ``S`` is a sum of positive semidefinite outer products. The correction is the square root of ``1 - \\lambda^{n_i}``, where a first-moment estimator divides by the first power.

The weights of ``S_{ij}`` sum to ``1 - \\lambda^{|\\mathcal{V}_i \\cap \\mathcal{V}_j|}``, and this is less than the denominator when the two assets have different histories. An entry for two such assets is therefore smaller in magnitude than the weighted mean over their common observations.

The location starts at zero and takes no correction. The first deviation of an asset is thus its return, and the next deviations read a location that the zero start damps.

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
  - [`EmpiricalPrior`](@ref)
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
        assert_unit_interval(decay, :decay)
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

Holds the running state of an incremental exponentially weighted covariance fit.

The struct is immutable, and each of its four fields is an array that the recursion writes in place. [`partial_fit!`](@ref) stores the state in the `cache` field of [`ExpWeightedCovariance`](@ref), and `cov(ce, state)` reads the estimate out of it.

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

Folds one observation into the state of an exponentially weighted covariance fit.

An asset is valid when its return is finite and the active mask admits it. An active asset with a non-finite return is on a holiday, and every entry of that asset keeps its value. An asset that leaves the active mask returns to the zero state, so its correction starts again if it lists again.

# Algorithm

 1. Mark as `valid` each asset whose return in `X` is finite and that `active_mask` admits.
 2. When `active_mask` is not `nothing`, take `newly_inactive` as the assets that the mask excludes and that `cache.active` admits. Set their rows and columns of `cache.covariance` and their counts in `cache.obs_count` to zero. Where `ce.centred` is `false`, set their entries of `cache.location` to `NaN`.
 3. Copy the mask into `cache.active`, or set every entry to `true` when the mask is `nothing`.
 4. Return `cache` when no asset is valid.
 5. Where `ce.centred` is `true`, take `Xi` as `X`. Otherwise read `cache.location` with `NaN` as zero into `loc`, move the valid entries of `cache.location` to `ce.decay * loc + (1 - ce.decay) * X`, and take `Xi` as `X - loc`.
 6. Take `e` as the entries of `Xi` of the valid assets, and move their block of `cache.covariance` to `ce.decay * block + (1 - ce.decay) * e * e'`.
 7. Add one to the count of each valid asset in `cache.obs_count`.

# Arguments

  - `cache::ExpWeightedCovarianceState`: State of the fit. The method writes it in place.
  - `ce::ExpWeightedCovariance`: Exponentially weighted covariance estimator.
  - `X::VecNum`: Returns of one observation, one entry per asset.
  - `active_mask::Option{<:AbstractVector{<:Bool}}`: Mask of the active assets at this observation. With `nothing` every asset is active, so a non-finite return is a holiday.

# Returns

  - `cache::ExpWeightedCovarianceState`: The state after the observation. It is the same object as the argument.

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

Covariance method of [`exp_weighted_pass!`](@ref). Runs one forward pass of the covariance recursion over the observations of `X`, and calls `f` after each observation.

# Algorithm

 1. Check `dims`, and check that `active_mask` has the size of `X` when it is not `nothing`.
 2. Take `N` as the number of assets.
 3. When `state` is `nothing`, make a zero state of `N` assets in the type `T` of a quotient of two entries of `X`. Its `covariance` and `obs_count` are zero, its `active` entries are `true`, and its `location` is zero where `est.centred` is `true` and `NaN` otherwise. When `state` is given, check that it holds `N` assets and take it as it is.
 4. For each observation `i`, fold the observation and its row of `active_mask` into the state with [`process_observation!`](@ref), then call `f(i, cache)`.
 5. Return the state.

# Arguments

  - `f`: Function called as `f(i, cache)` after the observation `i`. Its return value is discarded.
  - `est::ExpWeightedCovariance`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask::Option{<:AbstractMatrix{<:Bool}}`: Mask of the active assets, with the size of `X`. With `nothing` every asset is active.
  - `state::Option{<:ExpWeightedCovarianceState}`: State to continue. With `nothing` the pass starts from the zero state.

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`. A `DimensionMismatch` is thrown otherwise.
  - If `state` is not `nothing`, it holds as many assets as `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `cache::ExpWeightedCovarianceState`: The state after the last observation. A given `state` is the same object, written in place.

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

    # The state takes the type of a quotient of two returns, so an integer panel gets a
    # floating-point state and a `Float32` panel keeps a `Float32` state. An uncentred location
    # starts as `NaN`, which marks an asset with no valid observation, and the recursion reads
    # it as zero.
    T = typeof(one(eltype(X)) / one(eltype(X)))
    location = est.centred ? zeros(T, N) : fill(T(NaN), N)
    cache = if isnothing(state)
        ExpWeightedCovarianceState(zeros(T, N, N), location, zeros(Int, N), trues(N))
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

Covariance method of [`exp_weighted_pass!`](@ref). Runs one forward pass of the covariance recursion over the observations of `X`, and reads no intermediate state.

It calls the method that takes a callback, with a callback that does nothing.

# Arguments

  - `est::ExpWeightedCovariance`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask::Option{<:AbstractMatrix{<:Bool}}`: Mask of the active assets, with the size of `X`. With `nothing` every asset is active.
  - `state::Option{<:ExpWeightedCovarianceState}`: State to continue. With `nothing` the pass starts from the zero state.

# Validation

  - The checks of the method that takes a callback.

# Returns

  - `cache::ExpWeightedCovarianceState`: The state after the last observation.

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

Covariance method of [`exp_weighted_moment`](@ref). Reads the exponentially weighted covariance out of a state.

The method reads the state and does not write it, so a forward pass can call it after every observation. The struct docstring of [`ExpWeightedCovariance`](@ref) states the estimate that it computes.

# Algorithm

 1. Copy `cache.covariance` into `sigma`.
 2. Take `correction` as `1 / sqrt(max(1 - est.decay ^ n, eps(T)))` for each asset with a count `n` above zero, and as one for an asset with no valid observation. The `eps(T)` floor keeps the division finite.
 3. Multiply each entry `sigma[i, j]` by `correction[i] * correction[j]`.
 4. Take `not_ready` as the assets that `cache.active` excludes or whose count is below `est.min_obs`, and set their rows and columns of `sigma` to `NaN`.
 5. Replace the block of the other assets with the mean of the block and its transpose, which removes the round-off asymmetry.

# Arguments

  - `cache::ExpWeightedCovarianceState`: State to read.
  - `est::ExpWeightedCovariance`: Exponentially weighted covariance estimator.

# Returns

  - `sigma::MatNum`: Covariance matrix. The row and the column of an asset that is inactive, or that has fewer than `est.min_obs` valid observations, are `NaN`.

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

The struct docstring of [`ExpWeightedCovariance`](@ref) states the estimate.

# Algorithm

 1. Run one forward pass over the observations of `X` from the zero state with [`exp_weighted_pass!`](@ref), giving the state `cache`.
 2. Read `sigma` out of `cache` with [`exp_weighted_moment`](@ref).

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the size of `X`. An asset whose entry is `false` is inactive at that observation. Its row and column return to the zero state, and its answer is `NaN` while it stays inactive. With `nothing` every asset is active, so a non-finite return is a holiday, and every entry of that asset keeps its value.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `sigma::MatNum`: Covariance matrix of size `assets × assets`. The row and the column of an asset are `NaN` when the asset has fewer than `ce.min_obs` valid observations, or when `active_mask` excludes it at the last observation.

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
    return exp_weighted_moment(cache, ce)
end
"""
    gap_fill_value(ce::ExpWeightedCovariance) -> Float64

Answer `NaN`, so a gapped sample reaches the recursion with its gaps intact.

The recursion moves only the entries of the assets that are valid at an observation. A finite fill value would make a gapped asset valid, and the recursion would then move entries that a gap keeps. So the consumer passes the sample as it is, together with the active mask that explains each gap.

# Arguments

  - $(arg_dict[:ce])

# Returns

  - `fv::Float64`: `NaN`.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`gap_fill_value`](@ref)
  - [`Statistics.cov(ce::ExpWeightedCovariance, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function gap_fill_value(::ExpWeightedCovariance)
    return NaN
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

This is the covariance of the same call, rescaled to a unit diagonal. The correction of the covariance cancels in the rescale, so the correlation is the correlation of the internal state.

# Mathematical definition

```math
\\begin{align}
\\rho_{ij} &= \\max\\left(-1,\\, \\min\\left(1,\\, \\frac{S_{ij}}{\\sqrt{S_{ii} S_{jj}}}\\right)\\right)\\,, \\quad i \\neq j\\,, \\\\
\\rho_{ii} &= 1\\,.
\\end{align}
```

Where:

  - ``\\rho_{ij}``: Entry of the correlation matrix.
  - ``S_{ij}``: Entry of the internal state, as the struct docstring of [`ExpWeightedCovariance`](@ref) defines it.

The clamp changes an entry only when a holiday made the state indefinite. A clamped entry of ``\\pm 1`` is then not a correlation of the data. A blanked asset has `NaN` in its row, in its column and on the diagonal.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`.
  - $(arg_dict[:ignkwargs])

# Validation

  - The checks of the covariance of the same call.

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

Row `t` holds the diagonal of what `cov` returns for the first `t` observations of `X`, so no row reads an observation after its own. The recursion folds one observation at a time, so this method replaces the expanding-window fallback with one forward pass that reads the state after each observation.

The fallback cannot serve this estimator. It slices `X` once per row and passes every keyword as it is, so a mask of the whole window meets a window of `t` observations, and the size check refuses the call.

# Algorithm

 1. Check `dims`, and allocate `val` with one row per observation and one column per asset.
 2. Run one forward pass over `X` with [`exp_weighted_pass!`](@ref). After the observation `i`, write the diagonal of [`exp_weighted_moment`](@ref) of the state into row `i` of `val`.
 3. Return `val`, transposed when `dims == 2`.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if
    `dims == 2`. An asset is `NaN` at row `t` when it has fewer than `ce.min_obs` valid observations up to `t`, or when `active_mask` excludes it at `t`.

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

This estimator reads an active mask, so it replaces the reduce-and-expand root of the verb and reads the active mask of the panel itself. The answer covers the whole universe, not only the Coverage Universe. A young asset that lists inside the window gets an estimate from the observations it has, and it is `NaN` only while its count stays below `ce.min_obs`.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `pnl` is not `nothing`, its active mask has the size of `X` after the orientation that `dims` sets. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `sigma::MatNum`: Covariance matrix of size `assets × assets`.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
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

This is the covariance of the same call, rescaled to a unit diagonal, and it reads the active mask of the panel in the same way.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `pnl` is not `nothing`, its active mask has the size of `X` after the orientation that `dims` sets. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `rho::MatNum`: Correlation matrix of size `assets × assets`.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
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

This is the diagonal of the covariance of the same call, and it reads the active mask of the panel in the same way.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `pnl` is not `nothing`, its active mask has the size of `X` after the orientation that `dims` sets. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `var::MatNum`: Marginal variance, as a row where `dims` is `1` and as a column otherwise.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.var(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
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

This is the square root of the diagonal of the covariance of the same call, and it reads the active mask of the panel in the same way.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `pnl` is not `nothing`, its active mask has the size of `X` after the orientation that `dims` sets. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `std::MatNum`: Marginal volatility, as a row where `dims` is `1` and as a column otherwise.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`Statistics.var(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.std(ce::ExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel};
                        dims::Int = 1, kwargs...)
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
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

This is the diagonal of the covariance of the same call, read after each observation, and it reads the active mask of the panel in the same way. A Descriptor that holds this estimator thus keeps a number for a young asset. The reduce-and-expand root would reduce every window and answer `NaN` for it.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `pnl` is not `nothing`, its active mask has the size of `X` after the orientation that `dims` sets. A `DimensionMismatch` is thrown otherwise.

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
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
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

Fold a block of observations into the state that the estimator holds.

The estimator keeps the state in its `cache` field, so a second call continues the recursion and does not restart it. A partial-fit state is the one Result that an estimator can hold.

# Algorithm

 1. Run one forward pass over `X` with [`exp_weighted_pass!`](@ref), from `ce.cache` or from the zero state when `ce.cache` is `nothing`, giving `cache`.
 2. Return a copy of `ce` whose `cache` field is `cache`.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: Optional boolean matrix with the same size as `X`.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`. A `DimensionMismatch` is thrown otherwise.
  - If `ce.cache` is not `nothing`, it holds as many assets as `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `ce::ExpWeightedCovariance`: A new estimator whose `cache` holds the state after the block. When `ce.cache` was a state, the pass wrote that state in place, so the two estimators share it.

# Related

  - [`ExpWeightedCovariance`](@ref)
  - [`ExpWeightedCovarianceState`](@ref)
  - [`Statistics.cov(ce::ExpWeightedCovariance; kwargs...)`](@ref)
"""
function partial_fit!(ce::ExpWeightedCovariance{<:Any, <:Any, <:Any,
                                                <:Option{<:ExpWeightedCovarianceState}},
                      X::MatNum; dims::Int = 1,
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

Fold one observation into the state that the estimator holds.

The entries of `x` are the assets of one observation, which is one row of the matrix method. A caller that receives the observations one by one has this shape. The method folds `x` as a matrix of one row.

# Arguments

  - `ce`: Exponentially weighted covariance estimator.
  - `x::VecNum`: One observation, with one entry per asset.
  - `active_mask`: Optional boolean vector with the same length as `x`.
  - $(arg_dict[:ignkwargs])

# Validation

  - If `active_mask` is not `nothing`, `length(x) == length(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `x`.

# Returns

  - `ce::ExpWeightedCovariance`: A new estimator whose `cache` holds the state after the observation.

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
function partial_fit!(ce::ExpWeightedCovariance{<:Any, <:Any, <:Any,
                                                <:Option{<:ExpWeightedCovarianceState}},
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
    Statistics.cov(ce::ExpWeightedCovariance, state::ExpWeightedCovarianceState; kwargs...) -> MatNum

Read the exponentially weighted covariance out of a state that the caller holds.

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

Read the exponentially weighted covariance out of the state that the estimator holds.

The one-argument form reads the result of an incremental fit. [`partial_fit!`](@ref) leaves the state in the `cache` field, and this verb turns the state into the estimate. An estimator that has seen no observation has no state, so the call throws and does not answer zero.

# Arguments

  - `ce`: Exponentially weighted covariance estimator that holds a state.
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

Read the exponentially weighted correlation out of a state that the caller holds.

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

Read the exponentially weighted correlation out of the state that the estimator holds.

# Arguments

  - `ce`: Exponentially weighted covariance estimator that holds a state.
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

The fold ``S = \\lambda^{n_b} S_a + S_b`` is exact only for a centred estimator over a second block in which every asset is valid at every observation. An uncentred block reads the location that the first block carries. A holiday makes the decay of an entry depend on the joint observations of its two assets, and a reset discards the first block for its asset. The state records neither the holidays nor the resets, so a merge cannot tell the cases apart. Fold the second block into the first with `partial_fit!`.

# Arguments

  - `a`: State of the first block.
  - `b`: State of the second block.

# Validation

  - The method throws an `ArgumentError` for every pair.

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

It is the `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref) calls before it folds. It copies each of the four arrays.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::ExpWeightedCovarianceState`: A new state equal to `x`, with new arrays.

# Related

  - [`ExpWeightedCovarianceState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::ExpWeightedCovarianceState)
    return ExpWeightedCovarianceState(copy(x.covariance), copy(x.location),
                                      copy(x.obs_count), copy(x.active))
end

# An exponentially weighted recursion folds in every configuration (see
# [`supports_partial_fit`](@ref)).
function supports_partial_fit(::ExpWeightedCovariance)
    return true
end
export ExpWeightedCovariance
