"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all regime adjustment methods.

All concrete subtypes should subtype `RegimeAdjustedMethod` and implement the
[`regime_multiplier`](@ref) interface.

# Interfaces

In order to implement a new regime adjustment method that works seamlessly with the library,
subtype `RegimeAdjustedMethod` and implement the following method:

## `regime_multiplier` interface

  - `regime_multiplier(method::RegimeAdjustedMethod, regime_state::Number) -> Number`:
    Computes the variance scaling multiplier from the smoothed regime state.

### Arguments

  - `method`: The concrete regime adjustment method instance.
  - `regime_state::Number`: The current smoothed regime state value.

### Returns

  - `mult::Number`: The multiplicative scaling factor applied to the variance.

### Examples

```jldoctest
julia> struct MyRegimeMethod <: PortfolioOptimisers.RegimeAdjustedMethod end

julia> function PortfolioOptimisers.regime_multiplier(::MyRegimeMethod, s::Number)
           return abs(s)
       end

julia> PortfolioOptimisers.regime_multiplier(MyRegimeMethod(), -1.5)
1.5
```

## Related

  - [`LogRegimeAdjusted`](@ref)
  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RootMeanSquaredAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
abstract type RegimeAdjustedMethod <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Regime adjustment method that scales variance exponentially with the smoothed log-deviation
of standardised squared returns from its expected value under stationarity.

# Mathematical definition

The regime state ``s`` is defined as ``\\bar{s} = \\frac{1}{T}\\sum_t \\ln\\max(z_t^2, \\varepsilon) - \\kappa``, where
``\\kappa = \\psi(x) + \\ln y`` (``\\psi`` = digamma) is the stationary expectation of ``\\ln z^2`` under a
``\\chi^2(1)`` distribution scaled by ``y``, and ``\\varepsilon`` is a small positive threshold.

```math
\\begin{align}
\\kappa &= \\psi(x) + \\ln y\\,, \\\\
s &= \\frac{1}{T}\\sum_{t} \\ln\\!\\max(z_t^2, \\varepsilon) - \\kappa\\,, \\\\
\\mathrm{mult} &= \\exp(x \\cdot s)\\,.
\\end{align}
```

Where:

  - ``\\kappa``: Stationary expectation of ``\\ln z^2`` under the specified distribution.
  - ``\\psi``: Digamma function.
  - ``x``, ``y``: Parameters of [`LogRegimeAdjusted`](@ref).
  - ``s``: Smoothed log-deviation regime state.
  - $(math_dict[:T])
  - ``z_t^2``: Standardised squared return at time ``t``.
  - ``\\varepsilon``: Small positive threshold for numerical stability.
  - ``\\mathrm{mult}``: Variance scaling multiplier.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogRegimeAdjusted(;
        x::Number = 0.5,
        y::Number = 2.0
    ) -> LogRegimeAdjusted

Keywords correspond to the struct's fields. The `kappa` field is derived from `x` and `y`
and cannot be set directly.

## Validation

  - $(val_dict[:ra_x]) (i.e., `x >= 0`, finite, and non-empty).
  - $(val_dict[:ra_y]) (i.e., `y >= 0`, finite, and non-empty).

# Examples

```jldoctest
julia> LogRegimeAdjusted()
LogRegimeAdjusted
      x ┼ Float64: 0.5
      y ┼ Float64: 2.0
  kappa ┴ Float64: -1.2703628454614782
```

## Related

  - [`RegimeAdjustedMethod`](@ref)
  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RootMeanSquaredAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
@concrete struct LogRegimeAdjusted <: RegimeAdjustedMethod
    """
    $(field_dict[:ra_x])
    """
    x
    """
    $(field_dict[:ra_y])
    """
    y
    """
    $(field_dict[:ra_kappa])
    """
    kappa
    function LogRegimeAdjusted(x::Number, y::Number)
        assert_nonempty_nonneg_finite_val(x, :x)
        assert_nonempty_nonneg_finite_val(y, :y)
        kappa = SpecialFunctions.digamma(x) + log(y)
        return new{typeof(x), typeof(y), typeof(kappa)}(x, y, kappa)
    end
end
function LogRegimeAdjusted(; x::Number = 0.5, y::Number = 2.0)::LogRegimeAdjusted
    return LogRegimeAdjusted(x, y)
end
"""
$(DocStringExtensions.TYPEDEF)

Regime adjustment method that scales variance by the ratio of the mean absolute deviation
of standardised returns to the first-moment normalisation constant `x`.

# Mathematical definition

The regime state ``s`` and multiplier are:

```math
\\begin{align}
s &= \\frac{1}{x} \\cdot \\frac{1}{T}\\sum_t \\sqrt{\\max(z_t^2, 0)}\\,, \\\\
\\mathrm{mult} &= s\\,.
\\end{align}
```

Where:

  - ``s``: Regime state (ratio of mean absolute deviation to normalisation constant ``x``).
  - ``x = \\sqrt{2/\\pi}``: Expected value of ``|z|`` for a standard normal ``z``.
  - $(math_dict[:T])
  - ``z_t``: Standardised return at time ``t``.
  - ``\\mathrm{mult}``: Variance scaling multiplier.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FirstMomentRegimeAdjusted(;
        x::Number = sqrt(2 * inv(π))
    ) -> FirstMomentRegimeAdjusted

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ra_norm_x]) (i.e., `x >= 0`, finite, and non-empty).

# Examples

```jldoctest
julia> FirstMomentRegimeAdjusted()
FirstMomentRegimeAdjusted
  x ┴ Float64: 0.7978845608028654
```

## Related

  - [`RegimeAdjustedMethod`](@ref)
  - [`LogRegimeAdjusted`](@ref)
  - [`RootMeanSquaredAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
@concrete struct FirstMomentRegimeAdjusted <: RegimeAdjustedMethod
    """
    $(field_dict[:ra_norm_x])
    """
    x
    function FirstMomentRegimeAdjusted(x::Number)
        assert_nonempty_nonneg_finite_val(x, :x)
        return new{typeof(x)}(x)
    end
end
function FirstMomentRegimeAdjusted(;
                                   x::Number = sqrt(2 * inv(pi)))::FirstMomentRegimeAdjusted
    return FirstMomentRegimeAdjusted(x)
end
"""
$(DocStringExtensions.TYPEDEF)

Regime adjustment method that scales variance by the square root of the mean of the
standardised squared returns.

# Mathematical definition

```math
\\begin{align}
s &= \\frac{1}{T}\\sum_t z_t^2\\,, \\\\
\\mathrm{mult} &= \\sqrt{\\max(s, 0)}\\,.
\\end{align}
```

Where:

  - ``s``: Mean of standardised squared returns.
  - $(math_dict[:T])
  - ``z_t``: Standardised return at time ``t``.
  - ``\\mathrm{mult}``: Variance scaling multiplier.

## Related

  - [`RegimeAdjustedMethod`](@ref)
  - [`LogRegimeAdjusted`](@ref)
  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
struct RootMeanSquaredAdjusted <: RegimeAdjustedMethod end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the variance scaling multiplier for the log regime adjustment method.

# Arguments

  - `method::LogRegimeAdjusted`: Log regime adjustment method.
  - `regime_state::Number`: Current smoothed regime state.

# Returns

  - `mult::Number`: Variance scaling multiplier `exp(method.x * regime_state)`.

## Related

  - [`LogRegimeAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function regime_multiplier(method::LogRegimeAdjusted, regime_state::Number)
    return exp(method.x * regime_state)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the variance scaling multiplier for the first-moment regime adjustment method.

# Arguments

  - `::FirstMomentRegimeAdjusted`: First-moment regime adjustment method (unused).
  - `regime_state::Number`: Current smoothed regime state.

# Returns

  - `mult::Number`: Variance scaling multiplier equal to `regime_state` directly.

## Related

  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function regime_multiplier(::FirstMomentRegimeAdjusted, regime_state::Number)
    return regime_state
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the variance scaling multiplier for the root-mean-squared regime adjustment method.

# Arguments

  - `::RootMeanSquaredAdjusted`: Root-mean-squared regime adjustment method (unused).
  - `regime_state::Number`: Current smoothed regime state.

# Returns

  - `mult::Number`: Variance scaling multiplier `sqrt(max(regime_state, 0))`.

## Related

  - [`RootMeanSquaredAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function regime_multiplier(::RootMeanSquaredAdjusted, regime_state::Number)
    return sqrt(max(regime_state, zero(regime_state)))
end
"""
$(DocStringExtensions.TYPEDEF)

Online exponentially weighted variance estimator with regime-state adjustment.

At each observation, it updates a running exponentially weighted variance and computes
a standardised squared innovation `z²`. After accumulating enough observations, it
smooths a regime state using `regime_decay`, then scales the final variance by
`regime_multiplier(regime_method, regime_state)²`.

# Mathematical definition

EWM variance update (decay ``\\lambda``):

```math
\\begin{align}
v_t &= \\lambda v_{t-1} + (1 - \\lambda)(r_t - \\bar{r})^2\\,.
\\end{align}
```

Standardised innovation:

```math
\\begin{align}
z_t^2 &= (r_t - \\bar{r})^2 / v_t\\,.
\\end{align}
```

Regime state smoothed with `regime_decay` ``\\lambda_r`` (``g`` defined by [`RegimeAdjustedMethod`](@ref)):

```math
\\begin{align}
s_t &= \\lambda_r s_{t-1} + (1 - \\lambda_r) \\cdot g(z_t^2)\\,.
\\end{align}
```

Final variance:

```math
\\begin{align}
\\hat{\\sigma}^2 &= \\mathrm{mult}(s_T)^2 \\cdot v_T\\,.
\\end{align}
```

Where:

  - ``v_t``: Exponentially weighted variance at time ``t``.
  - ``\\lambda``: EWM decay parameter (`decay` field).
  - ``r_t``: Return at time ``t``.
  - ``\\bar{r}``: Mean return.
  - ``z_t^2``: Standardised squared innovation ``(r_t - \\bar{r})^2 / v_t``.
  - ``s_t``: Smoothed regime state at time ``t``.
  - ``\\lambda_r``: Regime decay parameter (`regime_decay` field).
  - ``g(\\cdot)``: Regime state transformation (see [`RegimeAdjustedMethod`](@ref)).
  - ``\\hat{\\sigma}^2``: Final regime-adjusted variance.
  - ``\\mathrm{mult}(s_T)``: Variance scaling multiplier (see [`RegimeAdjustedMethod`](@ref)).
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RegimeAdjustedExpWeightedVariance(;
        decay::Number             = exp2(-inv(40.0)),
        min_obs::Integer          = round(Int, max(1, inv(log2(inv(decay))))),
        hac_lags::Option{<:Integer} = nothing,
        regime_method::RegimeAdjustedMethod = FirstMomentRegimeAdjusted(),
        regime_decay::Number      = exp2(-2 / inv(log2(inv(decay)))),
        regime_min_obs::Integer   = round(Int, max(1, inv(log2(inv(decay))) / 2)),
        regime_lohi_mult::Option{<:Tuple{<:Number, <:Number}} = (0.7, 1.6),
        min_val::Number           = sqrt(eps()),
        centred::Bool             = false
    ) -> RegimeAdjustedExpWeightedVariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:decay])
  - $(val_dict[:min_obs])
  - $(val_dict[:hac_lags])
  - $(val_dict[:regime_min_obs])
  - $(val_dict[:regime_lohi_mult])

# Examples

```jldoctest
julia> ce = RegimeAdjustedExpWeightedVariance();

julia> ce.decay ≈ exp2(-inv(40.0))
true

julia> ce.min_obs
40
```

## Related

  - [`RegimeAdjustedMethod`](@ref)
  - [`LogRegimeAdjusted`](@ref)
  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RootMeanSquaredAdjusted`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
"""
@concrete struct RegimeAdjustedExpWeightedVariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    """
    $(field_dict[:hac_lags])
    """
    hac_lags
    """
    $(field_dict[:regime_method])
    """
    regime_method
    """
    $(field_dict[:regime_decay])
    """
    regime_decay
    """
    $(field_dict[:regime_min_obs])
    """
    regime_min_obs
    """
    $(field_dict[:regime_lohi_mult])
    """
    regime_lohi_mult
    """
    $(field_dict[:min_val])
    """
    min_val
    """
    $(field_dict[:centred])
    """
    centred
    function RegimeAdjustedExpWeightedVariance(decay::Number, min_obs::Integer,
                                               hac_lags::Option{<:Integer},
                                               regime_method::RegimeAdjustedMethod,
                                               regime_decay::Number,
                                               regime_min_obs::Integer,
                                               regime_lohi_mult::Option{<:Tuple{<:Number,
                                                                                <:Number}},
                                               min_val::Number, centred::Bool)
        assert_nonempty_gt0_finite_val(decay, :decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_nonempty_gt0_finite_val(regime_min_obs, :regime_min_obs)
        if !isnothing(regime_lohi_mult)
            @argcheck(zero(regime_lohi_mult[1]) < regime_lohi_mult[1] < regime_lohi_mult[2],
                      DomainError)
        end
        if !isnothing(hac_lags)
            assert_nonempty_gt0_finite_val(hac_lags, :hac_lags)
        end
        return new{typeof(decay), typeof(min_obs), typeof(hac_lags), typeof(regime_method),
                   typeof(regime_decay), typeof(regime_min_obs), typeof(regime_lohi_mult),
                   typeof(min_val), typeof(centred)}(decay, min_obs, hac_lags,
                                                     regime_method, regime_decay,
                                                     regime_min_obs, regime_lohi_mult,
                                                     min_val, centred)
    end
end
function RegimeAdjustedExpWeightedVariance(; decay::Number = exp2(-inv(40.0)),
                                           min_obs::Integer = round(Int,
                                                                    max(1,
                                                                        inv(log2(inv(decay))))),
                                           hac_lags::Option{<:Integer} = nothing,
                                           regime_method::RegimeAdjustedMethod = FirstMomentRegimeAdjusted(),
                                           regime_decay::Number = exp2(-2 /
                                                                       inv(log2(inv(decay)))),
                                           regime_min_obs::Integer = round(Int,
                                                                           max(1,
                                                                               inv(log2(inv(decay))) /
                                                                               2)),
                                           regime_lohi_mult::Option{<:Tuple{<:Number,
                                                                            <:Number}} = (0.7,
                                                                                          1.6),
                                           min_val::Number = sqrt(eps()),
                                           centred::Bool = false)::RegimeAdjustedExpWeightedVariance
    return RegimeAdjustedExpWeightedVariance(decay, min_obs, hac_lags, regime_method,
                                             regime_decay, regime_min_obs, regime_lohi_mult,
                                             min_val, centred)
end
"""
$(DocStringExtensions.TYPEDEF)

Internal mutable cache for the online variance update in [`RegimeAdjustedExpWeightedVariance`](@ref).

This type is an implementation detail and is not intended for direct use.

# Fields

$(DocStringExtensions.FIELDS)

## Related

  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
@concrete struct RegimeAdjustedVarianceCache <: AbstractResult
    """
    $(field_dict[:ret_buffer])
    """
    ret_buffer
    """
    $(field_dict[:ra_variance])
    """
    variance
    """
    $(field_dict[:ra_X2])
    """
    X2
    """
    $(field_dict[:ra_X_old_i])
    """
    X_old_i
    """
    $(field_dict[:ra_z2])
    """
    z2
    """
    $(field_dict[:ra_location])
    """
    location
    """
    $(field_dict[:obs_count])
    """
    obs_count
    """
    $(field_dict[:old_obs_count])
    """
    old_obs_count
    """
    $(field_dict[:ra_active])
    """
    active
    """
    $(field_dict[:regime_state])
    """
    regime_state
    """
    $(field_dict[:n_regime_obs])
    """
    n_regime_obs
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the scalar regime state for the root-mean-squared adjustment from valid
standardised squared innovations.

# Arguments

  - `::RootMeanSquaredAdjusted`: Root-mean-squared regime adjustment method (unused).
  - `z2_valid::VecNum`: Vector of valid (non-NaN) standardised squared innovations.
  - `::Any`: Ignored minimum value argument.

# Returns

  - `s::Number`: Mean of `z2_valid`.

## Related

  - [`RootMeanSquaredAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function get_regime_state(::RootMeanSquaredAdjusted, z2_valid::VecNum, ::Any)
    return Statistics.mean(z2_valid)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the scalar regime state for the first-moment adjustment from valid standardised
squared innovations.

# Arguments

  - `method::FirstMomentRegimeAdjusted`: First-moment regime adjustment method.
  - `z2_valid::VecNum`: Vector of valid (non-NaN) standardised squared innovations.
  - `::Any`: Ignored minimum value argument.

# Returns

  - `s::Number`: Mean absolute deviation `mean(sqrt(max.(z², 0))) / method.x`.

## Related

  - [`FirstMomentRegimeAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function get_regime_state(method::FirstMomentRegimeAdjusted, z2_valid::VecNum, ::Any)
    return Statistics.mean(sqrt.(max.(z2_valid, zero(eltype(z2_valid))))) / method.x
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the scalar regime state for the log adjustment from valid standardised squared
innovations.

# Arguments

  - `method::LogRegimeAdjusted`: Log regime adjustment method.
  - `z2_valid::VecNum`: Vector of valid (non-NaN) standardised squared innovations.
  - `min_val::Number`: Minimum threshold applied before taking logarithms.

# Returns

  - `s::Number`: Mean log deviation `mean(log(max.(z², min_val))) - method.kappa`.

## Related

  - [`LogRegimeAdjusted`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function get_regime_state(method::LogRegimeAdjusted, z2_valid::VecNum,
                          min_val::Number = sqrt(eps(eltype(z2_valid))))
    log_z2 = log.(max.(z2_valid, min_val))
    return Statistics.mean(log_z2) - method.kappa
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes (possibly HAC-corrected) squared returns for the current observation and stores
the result in `cache.X2`.

# Arguments

  - `cache::RegimeAdjustedVarianceCache`: Online variance computation cache.
  - `ce::RegimeAdjustedExpWeightedVariance`: Variance estimator configuration.
  - `X::VecNum`: Current centred returns vector.
  - `finite_mask::AbstractVector{<:Bool}`: Boolean mask of finite entries in `X`.

# Returns

  - `X2::VecNum`: The HAC-adjusted squared returns stored in `cache.X2`.

## Related

  - [`RegimeAdjustedVarianceCache`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function hac_squared_returns!(cache::RegimeAdjustedVarianceCache,
                              ce::RegimeAdjustedExpWeightedVariance, X::VecNum,
                              finite_mask::AbstractVector{<:Bool})
    copyto!(cache.X2, X .^ 2)
    if isnothing(cache.ret_buffer) || isempty(cache.ret_buffer)
        return cache.X2
    end

    for (i, X_old) in enumerate(Iterators.reverse(cache.ret_buffer))
        wi = one(eltype(X)) - i / (ce.hac_lags + 1)
        cache.X_old_i .= replace(X_old, NaN => zero(eltype(X_old)))
        cache.X2 .+= 2 * wi * X .* cache.X_old_i
    end
    cache.X2[finite_mask] .= max.(view(cache.X2, finite_mask), zero(eltype(cache.X2)))

    return cache.X2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Processes a single observation row (or column) to update the online variance cache.

Updates the running location, variance, and standardised squared innovations in `cache`,
then advances the smoothed regime state.

# Arguments

  - `cache::RegimeAdjustedVarianceCache`: Online variance computation cache (mutated).
  - `ce::RegimeAdjustedExpWeightedVariance`: Variance estimator configuration.
  - `X::VecNum`: Returns vector for the current observation.
  - `estimation_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask restricting which
    assets contribute to the regime state update.
  - `active_mask::Option{<:AbstractVector{<:Bool}}`: Optional mask of currently active
    assets. Inactive assets have their variance and counts reset.

# Returns

  - `nothing`.

## Related

  - [`RegimeAdjustedVarianceCache`](@ref)
  - [`RegimeAdjustedExpWeightedVariance`](@ref)
"""
function process_observation!(cache::RegimeAdjustedVarianceCache,
                              ce::RegimeAdjustedExpWeightedVariance, X::VecNum,
                              estimation_mask::Option{<:AbstractVector{<:Bool}},
                              active_mask::Option{<:AbstractVector{<:Bool}})
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
        return nothing
    end

    copyto!(cache.old_obs_count, cache.obs_count)

    Xi = if ce.centred
        X
    else
        loc = replace(cache.location, NaN => zero(eltype(cache.location)))
        cache.location[valid] = ce.decay * view(loc, valid) +
                                (one(eltype(cache.location)) - ce.decay) * view(X, valid)
        X - loc
    end

    regime_mask = valid .& (cache.old_obs_count .>= ce.min_obs)
    fill!(cache.z2, NaN)
    var_idx = regime_mask .& (cache.variance .>= ce.min_val)
    if any(var_idx)
        factor = inv.(max.(one(ce.decay) .- ce.decay .^ cache.old_obs_count[var_idx],
                           eps(ce.decay)))
        var_corrected = view(cache.variance, var_idx) .* factor
        cache.z2[var_idx] = view(Xi, var_idx) .^ 2 ./ var_corrected
    end

    X2 = hac_squared_returns!(cache, ce, Xi, valid)
    cache.variance[valid] .= ce.decay * view(cache.variance, valid) +
                             (one(ce.decay) - ce.decay) * view(X2, valid)
    cache.obs_count[valid] .+= 1

    if !isnothing(cache.ret_buffer)
        X_new = copy(Xi)
        X_new[.!valid] .= NaN
        push!(cache.ret_buffer, X_new)
    end

    if !isnothing(estimation_mask)
        regime_mask .&= estimation_mask
    end

    if !any(regime_mask)
        return nothing
    end

    z2_valid = filter(!isnan, view(cache.z2, regime_mask))
    if isempty(z2_valid)
        return nothing
    end

    regime_state = get_regime_state(ce.regime_method, z2_valid, ce.min_val)

    Accessors.@reset cache.regime_state = if isnothing(cache.regime_state)
        regime_state
    else
        ce.regime_decay * cache.regime_state +
        (one(eltype(ce.regime_decay)) - ce.regime_decay) * regime_state
    end

    Accessors.@reset cache.n_regime_obs += 1

    return nothing
end
"""
    Statistics.var(
        ce::RegimeAdjustedExpWeightedVariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Vector{<:Number}

Compute the regime-adjusted exponentially weighted variance for each asset.

Iterates over the observation dimension of `X`, updating an online variance cache at each
step. After processing all observations, applies a bias-correction factor and scales the
result by the square of the regime multiplier derived from the smoothed regime state.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted variance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix with the same size as `X`. When provided,
    only assets where `estimation_mask[i, :]` (or `[:, i]`) is `true` contribute to the
    regime state update for observation `i`.
  - `active_mask`: Optional boolean matrix with the same size as `X`. When provided,
    assets that become inactive have their variance and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `var::Vector{<:Number}`: Per-asset regime-adjusted exponentially weighted variance
    vector of length `features`. Assets with fewer than `ce.min_obs` observations return
    `NaN`.

## Related

  - [`RegimeAdjustedExpWeightedVariance`](@ref)
  - [`RegimeAdjustedVarianceCache`](@ref)
"""
function Statistics.var(ce::RegimeAdjustedExpWeightedVariance, X::MatNum; dims::Int = 1,
                        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be in (1, 2)"))
    est_flag = !isnothing(estimation_mask)
    act_flag = !isnothing(active_mask)
    itr, v = ifelse(isone(dims), (eachrow, (x, y) -> view(x, y, :)),
                    (eachcol, (x, y) -> view(x, :, y)))
    if est_flag
        @argcheck(size(X) == size(estimation_mask),
                  DimensionMismatch("size(X) ($(size(X))) must match size(estimation_mask) ($(size(estimation_mask)))"))
    end
    if act_flag
        @argcheck(size(X) == size(active_mask),
                  DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(active_mask)))"))
    end
    N = size(X, setdiff((1, 2), (dims,))[1])

    cache = RegimeAdjustedVarianceCache(if isnothing(ce.hac_lags)
                                            nothing
                                        else
                                            DataStructures.CircularBuffer{Vector{eltype(X)}}(ce.hac_lags)
                                        end, zeros(eltype(X), N), zeros(eltype(X), N),
                                        zeros(eltype(X), N), fill(NaN, N),
                                        ce.centred ? zeros(eltype(X), N) : fill(NaN, N),
                                        zeros(Int, N), zeros(Int, N), trues(N), nothing,
                                        zero(eltype(X)))
    for (i, Xi) in enumerate(itr(X))
        emi = est_flag ? v(estimation_mask, i) : nothing
        ami = act_flag ? v(active_mask, i) : nothing
        process_observation!(cache, ce, Xi, emi, ami)
    end

    variance = copy(cache.variance)
    correction = ones(eltype(variance), length(variance))
    correction[cache.obs_count .> zero(eltype(cache.obs_count))] .= inv.(max.(one(ce.decay) .-
                                                                              ce.decay .^
                                                                              cache.obs_count,
                                                                              eps(eltype(variance))))
    variance .*= correction
    not_ready = .!cache.active .| (cache.obs_count .< ce.min_obs)

    if any(not_ready)
        variance[not_ready] .= NaN
    end

    if !ce.centred && any(.!cache.active)
        cache.location[.!cache.active] .= NaN
    end

    factor = if cache.n_regime_obs < ce.regime_min_obs
        one(eltype(X))
    else
        regime_multiplier(ce.regime_method, cache.regime_state)
    end

    return variance * factor^2
end

export LogRegimeAdjusted, FirstMomentRegimeAdjusted, RootMeanSquaredAdjusted,
       RegimeAdjustedExpWeightedVariance
