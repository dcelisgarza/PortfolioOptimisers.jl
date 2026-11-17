"""
$(DocStringExtensions.TYPEDEF)

The carried state of the recursion of an [`EWMacroSensitivity`](@ref), after the observations that it folded.

The state holds the [`EWBetaState`](@ref) of the regression of each asset on the market and the reference series, the flag of each asset whose market return has not changed over its observations, and the two counts that the refusal of a gap in a named reference series reads: the windows whose reference return held a finite value, and whether the open window holds one yet.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWMacroSensitivityState(; be::EWBetaState, cm::AbstractVector{Bool}, c::Integer,
                            hf::Bool) -> EWMacroSensitivityState

Keywords correspond to the struct's fields.

## Validation

  - `cm` holds one entry per asset of `be`, and the moments of `be` track three series. A `DimensionMismatch` is thrown otherwise.

# Related

  - [`EWMacroSensitivity`](@ref)
  - [`EWBetaState`](@ref)
  - [`ew_macro_beta`](@ref)
  - [`ew_beta_fold`](@ref)
"""
@concrete struct EWMacroSensitivityState <: AbstractPartialFitState
    """
    The [`EWBetaState`](@ref) of the regression of each asset on the market and on the reference series.
    """
    be
    """
    The flag of each asset whose market return has taken one value over every observation of its current history. The intercept absorbs such a market, so the partial beta is the slope on the reference series alone.
    """
    cm
    """
    The count of the windows whose reference return held a finite value.
    """
    c
    """
    Whether the window that is not yet complete holds a finite reference return.
    """
    hf
    function EWMacroSensitivityState(be::EWBetaState, cm::AbstractVector{Bool}, c::Integer,
                                     hf::Bool)
        @argcheck(length(cm) == length(be.b) && length(be.y) == 3,
                  DimensionMismatch("the state of an EWMacroSensitivity holds one constant-market flag per asset and the moments of three series, got $(length(cm)) flags, $(length(be.b)) assets and $(length(be.y)) series"))
        return new{typeof(be), typeof(cm), typeof(c), typeof(hf)}(be, cm, c, hf)
    end
end
function EWMacroSensitivityState(; be::EWBetaState, cm::AbstractVector{Bool}, c::Integer,
                                 hf::Bool)::EWMacroSensitivityState
    return EWMacroSensitivityState(be, cm, c, hf)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`EWMacroSensitivityState`](@ref), so that the copy shares no array with the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::EWMacroSensitivityState`: A new state, equal to `x`.

# Related

  - [`EWMacroSensitivityState`](@ref)
"""
function Base.copy(x::EWMacroSensitivityState)
    return EWMacroSensitivityState(copy(x.be), copy(x.cm), x.c, x.hf)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`EWMacroSensitivityState`](@ref): the recursion of the second block starts from the state after the first one, so two states fitted on disjoint blocks do not give the state of their union to the last bit.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`EWMacroSensitivityState`](@ref)
  - [`partial_fit!`](@ref)
"""
function merge_states(::EWMacroSensitivityState, ::EWMacroSensitivityState)
    return throw(ArgumentError("an EWMacroSensitivityState cannot merge two states fitted on disjoint blocks: the recursion of the second block starts from the state after the first one. Fold the second block into the state of the first with partial_fit!."))
end
"""
    ew_macro_beta(mo::EWMoments, cm::Bool, i::Integer, min_obs::Integer) -> Real

Returns the partial beta of asset `i` on the reference series after the market is removed, from its moments, as [`EWMacroSensitivity`](@ref) states.

Where the market has not changed over the observations of the asset, the intercept absorbs it, and the partial beta is the slope of [`ew_macro_slope`](@ref) on the reference series alone. Otherwise the Frisch-Waugh form of [`ew_macro_partial`](@ref) removes the market. Each of the two states when its value is defined, and the partial beta is `NaN` where it is not, or where the quotient is not finite.

# Arguments

  - `mo`: The moments of the market, of the reference series and of the asset.
  - `cm`: The constant-market flag of the asset.
  - `i`: The asset.
  - $(arg_dict[:min_obs])

# Returns

  - `b::Real`: The partial beta, or `NaN`.

# Related

  - [`ew_macro_slope`](@ref)
  - [`ew_macro_partial`](@ref)
  - [`EWMacroSensitivity`](@ref)
  - [`ew_regression_series!`](@ref)
"""
function ew_macro_beta(mo::EWMoments, cm::Bool, i::Integer, min_obs::Integer)::Real
    b, ok = cm ? ew_macro_slope(mo, i, min_obs) : ew_macro_partial(mo, i, min_obs)
    return ok && isfinite(b) ? b : oftype(b, NaN)
end
"""
    ew_macro_slope(mo::EWMoments, i::Integer, min_obs::Integer) -> Tuple{Real, Bool}

Returns the slope of asset `i` on the reference series alone, the partial beta where the market is constant over the observations of the asset, and whether it is defined.

The slope is the covariance numerator of the reference series with the asset over that of the reference series with itself. It is defined from `max(2, min_obs)` observations, where the reference series varies as [`ew_moments_vary`](@ref) states.

# Arguments

  - `mo`: The moments of the market, of the reference series and of the asset.
  - `i`: The asset.
  - $(arg_dict[:min_obs])

# Returns

  - `(b, ok)::Tuple{Real, Bool}`: The plain tuple of the slope and of whether it is defined.

# Related

  - [`ew_macro_beta`](@ref)
"""
function ew_macro_slope(mo::EWMoments, i::Integer, min_obs::Integer)::Tuple{Real, Bool}
    ok = mo.n[i] >= max(2, min_obs) && ew_moments_vary(mo, 2, i)
    return mo.C[2, 3, i] / mo.C[2, 2, i], ok
end
"""
    ew_macro_partial(mo::EWMoments, i::Integer, min_obs::Integer) -> Tuple{Real, Bool}

Returns the Frisch-Waugh partial beta of asset `i` on the reference series after the market, from the covariance numerators of its three series, and whether it is defined.

The computation works in standardised regressors, so it forms no product of two variances, which could leave the range of the arithmetic for a very large or a very small return. The partial beta is defined from `max(3, min_obs)` observations, where the market and the reference series both vary as [`ew_moments_vary`](@ref) states, and where the two are far enough from collinear.

# Mathematical definition

```math
\\begin{align}
\\beta^{f}_{t,i} &= \\frac{1}{\\sqrt{S^{ff}}} \\, \\frac{S^{fx} / \\sqrt{S^{ff}} - \\rho\\, S^{mx} / \\sqrt{S^{mm}}}{1 - \\rho^2}\\,, \\\\
\\rho &= \\frac{S^{mf}}{\\sqrt{S^{mm}}\\, \\sqrt{S^{ff}}}\\,.
\\end{align}
```

Where:

  - ``\\beta^{f}_{t,i}``: Partial beta of asset ``i`` on the reference series.
  - ``S^{mm}``, ``S^{ff}``, ``S^{mf}``: Covariance numerators of the market, of the reference series, and of the two, over the observations of asset ``i``.
  - ``S^{mx}``, ``S^{fx}``: Covariance numerators of the market and of the reference series with the return of asset ``i``.
  - ``\\rho``: Weighted correlation of the market with the reference series.

``1 - \\rho^2`` is computed as ``(1 - |\\rho|)(1 + |\\rho|)``, the share of the variance of the reference series that the market leaves, and the regressors pass where it is at least `1e-8`.

# Arguments

  - `mo`: The moments of the market, of the reference series and of the asset.
  - `i`: The asset.
  - $(arg_dict[:min_obs])

# Returns

  - `(b, ok)::Tuple{Real, Bool}`: The plain tuple of the partial beta and of whether it is defined.

# Related

  - [`ew_macro_beta`](@ref)
"""
function ew_macro_partial(mo::EWMoments, i::Integer, min_obs::Integer)::Tuple{Real, Bool}
    C = mo.C
    sm = sqrt(C[1, 1, i])
    sf = sqrt(C[2, 2, i])
    rho = (C[1, 2, i] / sm) / sf
    rf = (1 - abs(rho)) * (1 + abs(rho))
    rc = C[2, 3, i] / sf - rho * (C[1, 3, i] / sm)
    ok = rf >= 1e-8 &&
         mo.n[i] >= max(3, min_obs) &&
         ew_moments_vary(mo, 1, i) &&
         ew_moments_vary(mo, 2, i)
    return (rc / rf) / sf, ok
end
"""
    ew_model_reset!(cm::AbstractVector{Bool}, i::Integer) -> nothing
    ew_model_update!(cm::AbstractVector{Bool}, mo::EWMoments, y::AbstractVector{<:Real},
                     i::Integer) -> nothing
    ew_model_beta(cm::AbstractVector{Bool}, mo::EWMoments, i::Integer,
                  min_obs::Integer) -> Real

The model of [`ew_regression_series!`](@ref) for an [`EWMacroSensitivity`](@ref): the constant-market flag of each asset.

A reset sets the flag of the asset, because a history of no observation has a constant market. An update keeps the flag only where the asset has no observation yet or the new market value equals the last one, and it runs before the moments take the value. The beta is [`ew_macro_beta`](@ref).

# Arguments

  - `cm`: The constant-market flags, changed in place.
  - `mo`: The moments.
  - `y`: The values of the series, the market return first.
  - `i`: The asset.
  - $(arg_dict[:min_obs])

# Returns

  - `nothing`, or the partial beta `b::Real` for `ew_model_beta`.

# Related

  - [`ew_regression_series!`](@ref)
  - [`EWMacroSensitivityState`](@ref)
"""
function ew_model_reset!(cm::AbstractVector{Bool}, i::Integer)::Nothing
    cm[i] = true
    return nothing
end
function ew_model_update!(cm::AbstractVector{Bool}, mo::EWMoments,
                          y::AbstractVector{<:Real}, i::Integer)::Nothing
    cm[i] &= iszero(mo.n[i]) || y[1] == mo.r[1, i]
    return nothing
end
function ew_model_beta(cm::AbstractVector{Bool}, mo::EWMoments, i::Integer,
                       min_obs::Integer)::Real
    return ew_macro_beta(mo, cm[i], i, min_obs)
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted sensitivity of the returns to a reference series, after the market is removed.

The exposure of an asset to an exchange rate, a rate of interest, inflation or a basket of commodities is the part of its move that the market does not explain. The partial beta is the coefficient of the reference series in the regression of the returns on the market and the reference series together, so it contains no exposure that the market explains. The reference series belongs to no asset, so it is not a Panel Field. The field `series` names its column in the Exogenous Series `rd.E` of the returns data, and [`descriptor`](@ref) reads that column. A [`CrossSectionalFactorPrior`](@ref) and a cross-validation fold pass the returns data alone, so inside them the field is the only path. A direct call can pass the series as the keyword `ref` instead.

# Mathematical definition

```math
\\begin{align}
x_{t,\\,i} &= \\alpha_i + \\beta^{m}_i\\, r_{m,t} + \\beta^{f}_i\\, r_{f,t} + \\varepsilon_{t,i}\\,, \\\\
\\beta^{f}_{t,i} &= \\frac{S^{fx}_{t,i} - S^{mx}_{t,i}\\, S^{mf}_{t,i} / S^{mm}_{t,i}}{S^{ff}_{t,i} - \\left(S^{mf}_{t,i}\\right)^2 / S^{mm}_{t,i}}\\,.
\\end{align}
```

Where:

  - ``r_{f,t}``: Reference return at observation ``t``.
  - ``\\alpha_i``, ``\\beta^{m}_i``, ``\\varepsilon_{t,i}``: Intercept, market beta and residual of asset ``i`` in the regression.
  - ``\\beta^{f}_i``: Partial sensitivity of asset ``i`` to the reference series.
  - ``\\beta^{f}_{t,i}``: The Descriptor, the exponentially weighted estimate of ``\\beta^{f}_i`` after observation ``t``. It is the Frisch-Waugh form of the coefficient.
  - ``S^{mm}_{t,i}``, ``S^{ff}_{t,i}``, ``S^{mf}_{t,i}``, ``S^{mx}_{t,i}``, ``S^{fx}_{t,i}``: Covariance numerators of the market, of the reference series and of the return of asset ``i`` with one another, over the valid observations of asset ``i``, from [`EWMoments`](@ref).
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:r_mt_ewb])

Every sum runs over the observations of the asset alone where its return, the market return and the reference return are all finite and the asset is active, with the weights that [`EWBeta`](@ref) states. An observation whose reference return is missing therefore leaves every asset as it was. An inactive observation resets the asset. [`ew_macro_beta`](@ref) computes the partial beta in standardised regressors, and it is `NaN` where a regressor does not vary or the two are almost perfectly correlated. A market that does not change over the observations of an asset is absorbed by the intercept, and the partial beta is then the slope on the reference series alone. A series that `series` names is refused where a window it reads after the warm-up holds no finite value, as [`ew_macro_windows`](@ref) states. An infinite reference return is refused on either path.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWMacroSensitivity(; mcap::AbstractString = "market_cap",
                       series::Option{<:AbstractString} = nothing, half_life::Real = 60.0,
                       decay::Real = half_life_decay(half_life),
                       min_obs::Integer = half_life_min_obs(half_life),
                       agg_obs::Integer = 1,
                       cache::Option{<:AbstractPartialFitState} = nothing) -> EWMacroSensitivity

Keywords correspond to the struct's fields, except `half_life`, which is not a field. It sets the defaults of `decay` and `min_obs`, and a value passed for either of those is used as it is. With daily observations, the default half-life of `60` is about one quarter of a year. The default `series = nothing` names no column, so a call to [`descriptor`](@ref) must then pass the keyword `ref`.

## Validation

  - If `series` is not `nothing`, `!isempty(series)`.
  - $(val_dict[:decay])
  - $(val_dict[:min_obs])
  - `agg_obs >= 1`.

# Examples

```jldoctest
julia> EWMacroSensitivity(; series = \"EURUSD\", half_life = 2)
EWMacroSensitivity
     mcap ┼ String: "market_cap"
   series ┼ String: "EURUSD"
    decay ┼ Float64: 0.7071067811865476
  min_obs ┼ Int64: 2
  agg_obs ┴ Int64: 1
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWBeta`](@ref)
  - [`ew_macro_beta`](@ref)
  - [`ew_macro_reference`](@ref)
  - [`market_return_series`](@ref)
"""
@concrete struct EWMacroSensitivity <: AbstractDescriptorEstimator
    """
    $(field_dict[:mcap])
    """
    mcap
    """
    Name of the column of the Exogenous Series `rd.E` that holds the reference return, or `nothing` when a call to [`descriptor`](@ref) passes the series as the keyword `ref`.
    """
    series
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    """
    $(field_dict[:agg_obs])
    """
    agg_obs
    """
    $(field_dict[:ew_beta_desc_cache])
    """
    cache
    function EWMacroSensitivity(mcap::AbstractString, series::Option{<:AbstractString},
                                decay::Real, min_obs::Integer, agg_obs::Integer,
                                cache::Option{<:AbstractPartialFitState})
        assert_panel_terms(mcap, :mcap)
        @argcheck(isnothing(series) || !isempty(series),
                  IsEmptyError("series names a column of the Exogenous Series, so it cannot be the empty string"))
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_ew_agg_obs(agg_obs)
        return new{typeof(mcap), typeof(series), typeof(decay), typeof(min_obs),
                   typeof(agg_obs), typeof(cache)}(mcap, series, decay, min_obs, agg_obs,
                                                   cache)
    end
end
function EWMacroSensitivity(; mcap::AbstractString = "market_cap",
                            series::Option{<:AbstractString} = nothing,
                            half_life::Real = 60.0,
                            decay::Real = half_life_decay(half_life),
                            min_obs::Integer = half_life_min_obs(half_life),
                            agg_obs::Integer = 1,
                            cache::Option{<:AbstractPartialFitState} = nothing)::EWMacroSensitivity
    return EWMacroSensitivity(mcap, series, decay, min_obs, agg_obs, cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Holds the values that a step of an [`EWMacroSensitivity`](@ref) reads off its carried state to check its reference return.

This type is an implementation detail and is not intended for direct use. [`ew_macro_carried`](@ref) makes it, all at zero for the batch call, and [`ew_macro_windows`](@ref) and [`ew_beta_fold`](@ref) read it. `c` and `t0` are integers of one type, so a record names them where a plain tuple would let a swap pass silently.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ew_macro_carried`](@ref)
  - [`ew_macro_windows`](@ref)
"""
@concrete struct CarriedReference
    """
    `c`: The count of the windows whose reference return held a finite value.
    """
    c
    """
    `hf`: Whether the open window holds a finite reference return.
    """
    hf
    """
    `t0`: The number of observations that the state folded.
    """
    t0
end
"""
    ew_macro_reference(series::Nothing, de::EWMacroSensitivity, rd::ReturnsResult,
                       ref::Option{<:AbstractVector{<:Real}}) -> AbstractVector{<:Real}
    ew_macro_reference(series::AbstractString, de::EWMacroSensitivity, rd::ReturnsResult,
                       ref::Option{<:AbstractVector{<:Real}}) -> AbstractVector{<:Real}

Select the reference return of an [`EWMacroSensitivity`](@ref), from the keyword `ref` or from the column of the Exogenous Series that the field `series` names.

The field and the keyword are two sources for one series, so a call that gives both is refused. The ingestion pads the Exogenous Series with `NaN` where the series is silent, for example before the first value of a series that starts late, and [`ew_macro_windows`](@ref) checks those gaps.

# Algorithm

 1. With `series = nothing`, refuse a `ref` that is `nothing` or whose length is not the number of observations, and return `ref`.
 2. Otherwise, refuse a `ref` that is not `nothing`, an `rd.E` that is `nothing`, and a `series` that `rd.ne` does not hold. Return the column of `rd.E` that `series` names.

# Arguments

  - `series`: The field `series` of `de`.
  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd])
  - `ref`: The keyword `ref` of [`descriptor`](@ref).

# Validation

  - With `series = nothing`, `ref` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - With `series = nothing`, `length(ref) == size(rd.X, 1)`. Raises a `DimensionMismatch`.
  - With a `series`, `ref` is `nothing`. Raises an `ArgumentError`.
  - With a `series`, `rd.E` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - With a `series`, `rd.ne` holds `series`. Raises an `ArgumentError` that names the series.

# Returns

  - `rf::AbstractVector{<:Real}`: The reference return, one entry per observation.

# Related

  - [`EWMacroSensitivity`](@ref)
  - [`descriptor`](@ref)
  - [`ew_macro_windows`](@ref)
  - [`ReturnsResult`](@ref)
"""
function ew_macro_reference(::Nothing, ::EWMacroSensitivity, rd::ReturnsResult,
                            ref::Option{<:AbstractVector{<:Real}})::AbstractVector{<:Real}
    @argcheck(!isnothing(ref),
              IsNothingError("a macro sensitivity is measured against a reference return series, and the estimator names no column of the Exogenous Series in `series`, and the call passes no keyword `ref`. Set `series` to the name of a column of rd.E, or pass the series as `ref`, one entry per observation."))
    @argcheck(length(ref) == size(rd.X, 1),
              DimensionMismatch("the reference return series carries one entry per observation, got length(ref) = $(length(ref)) and size(rd.X, 1) = $(size(rd.X, 1))"))
    return ref
end
function ew_macro_reference(series::AbstractString, ::EWMacroSensitivity, rd::ReturnsResult,
                            ref::Option{<:AbstractVector{<:Real}})::AbstractVector{<:Real}
    @argcheck(isnothing(ref),
              ArgumentError("the estimator reads its reference return from the column \"$series\" of the Exogenous Series, and the call also passes the keyword `ref`. Give the series in one place: set `series = nothing` to pass `ref`, or do not pass `ref`."))
    @argcheck(!isnothing(rd.E),
              IsNothingError("the estimator reads its reference return from the column \"$series\" of the Exogenous Series of the returns data, and rd.E is nothing. Give the PricesResult or the ReturnsResult an E with a column named \"$series\"."))
    j = findfirst(==(series), rd.ne)
    @argcheck(!isnothing(j),
              ArgumentError("the estimator reads its reference return from the column \"$series\" of the Exogenous Series, and rd.ne has no column of that name. Got rd.ne => $(rd.ne)"))
    return view(rd.E, :, j)
end
"""
    ew_macro_windows(series::Option{<:AbstractString}, rf::AbstractVector{<:Real},
                     de::EWMacroSensitivity, carried::CarriedReference) -> Tuple{Int, Bool}

Walk the windows of `agg_obs` observations of the reference return, count those that hold a finite value, and refuse a gap in a named series after the warm-up.

The recursion reads a window only where it holds a finite reference return, and it skips the other windows. In the warm-up that is correct. After the warm-up, a skipped window freezes the partial beta of every asset, so a series that `series` names is refused where a window holds no finite value there. A window with a gap and at least one finite value is accepted. The keyword `ref` keeps the rule of the recursion at every window. The warm-up ends when `min_obs` windows have held a finite value. A step of the carry fold starts from the counts and the number of observations that its state carries, so it refuses the gap that the batch call refuses and names the same observations.

# Arguments

  - `series`: The field `series` of `de`.
  - `rf`: The reference return of the step, one entry per observation.
  - `de`: Descriptor Estimator.
  - `carried`: The values that [`ew_macro_carried`](@ref) reads off the carried state, all at zero for the batch call.

# Validation

  - With a `series`, every window of `rf` after the warm-up holds a finite value. Raises an [`IsNonFiniteError`](@ref) that names the series and the observations of the window.

# Returns

  - `(c, hf)::Tuple{Int, Bool}`: The plain tuple of the count of the windows with a finite value, and of whether the open window holds one, after the step.

# Related

  - [`EWMacroSensitivity`](@ref)
  - [`ew_macro_reference`](@ref)
  - [`ew_macro_carried`](@ref)
"""
function ew_macro_windows(series::Option{<:AbstractString}, rf::AbstractVector{<:Real},
                          de::EWMacroSensitivity, carried::CarriedReference)
    (; c, hf, t0) = carried
    a = de.agg_obs
    for (j, f) in enumerate(rf)
        hf |= isfinite(f)
        if iszero((t0 + j) % a)
            ew_macro_refuse(series, c >= de.min_obs && !hf, a, t0 + j)
            c += hf
            hf = false
        end
    end
    return c, hf
end
"""
    ew_macro_refuse(::Nothing, ::Bool, ::Integer, ::Integer) -> nothing
    ew_macro_refuse(series::AbstractString, gap::Bool, a::Integer, t::Integer) -> nothing

Refuses the window of `a` observations that closes at observation `t` where it is a `gap` of a named reference series, as [`ew_macro_windows`](@ref) states. The keyword `ref` names no series, and it refuses nothing.

# Arguments

  - `series`: The field `series` of the estimator.
  - `gap`: Whether the window holds no finite value after the warm-up.
  - `a`: The number of observations of a window.
  - `t`: The observation that closes the window.

# Validation

  - A named series has no `gap`. Raises an [`IsNonFiniteError`](@ref).

# Returns

  - `nothing`.

# Related

  - [`ew_macro_windows`](@ref)
"""
function ew_macro_refuse(::Nothing, ::Bool, ::Integer, ::Integer)::Nothing
    return nothing
end
function ew_macro_refuse(series::AbstractString, gap::Bool, a::Integer, t::Integer)::Nothing
    if gap
        at = isone(a) ? "at observation $t" : "at observations $(t - a + 1) to $t"
        throw(IsNonFiniteError("the reference return \"$series\" must hold a finite value in every window of $a observation(s) that the estimator reads after its warm-up, and it holds none $at of the returns data. A gap in the warm-up is accepted, because the recursion skips it before it writes a sensitivity. Fill the gap, or pass the series as the keyword `ref`, whose recursion holds its state there."))
    end
    return nothing
end
"""
    descriptor(de::EWMacroSensitivity, rd::ReturnsResult;
               ref::Option{<:AbstractVector{<:Real}} = nothing) -> Matrix{<:Real}

Compute an exponentially weighted macro sensitivity Descriptor from a [`ReturnsResult`](@ref) and a reference series.

The reference series is the column of the Exogenous Series `rd.E` that the field `series` of `de` names. A direct call can pass the series as the keyword `ref` instead, when `series` is `nothing`.

# Algorithm

 1. Select the reference return `rf` through [`ew_macro_reference`](@ref), refuse an infinite value of it on any observation, and check its windows through [`ew_macro_windows`](@ref).
 2. Build the market return `rm` through [`market_return_series`](@ref).
 3. Run the regression of each asset on the market and on `rf` from a new state through [`ew_regression_series!`](@ref), with the constant-market flags of [`EWMacroSensitivityState`](@ref) as its model.
 4. Write `NaN` into the inactive cells through [`descriptor_active_fill!`](@ref).

[`ew_beta_fold`](@ref) holds these steps, and a step of the carry fold runs them from the carried state.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`, and the Exogenous Series in `rd.ne` and `rd.E` when `de.series` names a column.
  - `ref`: The reference return, one entry per observation. It is required when `de.series` is `nothing`, and refused otherwise.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - The rules of [`ew_macro_reference`](@ref) and of [`ew_macro_windows`](@ref).
  - No value of the reference return is infinite. `NaN` is a gap, and an infinite value is not. Raises an [`IsNonFiniteError`](@ref) that names the observation.
  - The rules of [`market_return_series`](@ref).
  - The moments stay finite. Raises an [`IsNonFiniteError`](@ref).

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 2.0; 3.0 4.0; 5.0 6.0; 2.0 1.0])];
                         amsk = trues(4, 2), emsk = trues(4, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05; 0.02 -0.03],
                          ne = [\"EURUSD\"], E = reshape([0.02, -0.01, 0.03, 0.01], :, 1), pnl = pnl);

julia> D = descriptor(EWMacroSensitivity(; series = \"EURUSD\", half_life = 1), rd)
4×2 Matrix{Float64}:
 NaN        NaN
 NaN        NaN
   2.29787   -1.44681
   2.22631   -1.36007

julia> isequal(D,
               descriptor(EWMacroSensitivity(; half_life = 1), rd;
                          ref = [0.02, -0.01, 0.03, 0.01]))
true
```

# Related

  - [`EWMacroSensitivity`](@ref)
  - [`ew_macro_reference`](@ref)
  - [`ew_macro_beta`](@ref)
  - [`ew_beta_fold`](@ref)
  - [`market_return_series`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function descriptor(de::EWMacroSensitivity, rd::ReturnsResult;
                    ref::Option{<:AbstractVector{<:Real}} = nothing)::Matrix{<:Real}
    return ew_beta_fold(de, rd, nothing; ref = ref).D
end
"""
$(DocStringExtensions.TYPEDEF)

The carried state of the recursion of an [`EWDownsideBeta`](@ref), after the observations that it folded.

The state holds every quantity that [`ew_downside_beta_series!`](@ref) reads from the observation before: the co-moment of each asset with the market, the second moment of the shortfall of the market over the observations of each asset, and the valid count of each asset. A step of the carry fold of a [`CrossSectionalFactorPrior`](@ref) folds its new observations from it with the arithmetic of the batch call.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWDownsideBetaState(; cd::AbstractVector{<:Real}, vd::AbstractVector{<:Real},
                        n::AbstractVector{<:Integer}) -> EWDownsideBetaState

Keywords correspond to the struct's fields.

## Validation

  - `cd`, `vd` and `n` hold the same number of assets. A `DimensionMismatch` is thrown otherwise.

# Related

  - [`ew_downside_beta_series!`](@ref)
  - [`ew_beta_fold`](@ref)
  - [`descriptor_step`](@ref)
"""
@concrete struct EWDownsideBetaState <: AbstractPartialFitState
    """
    The co-moment ``C^{-}_{t,i}`` of each asset with the market.
    """
    cd
    """
    The second moment ``V^{-}_{t,i}`` of the shortfall of the market, over the valid observations of each asset.
    """
    vd
    """
    The count of the valid observations of each asset since its last reset.
    """
    n
    function EWDownsideBetaState(cd::AbstractVector{<:Real}, vd::AbstractVector{<:Real},
                                 n::AbstractVector{<:Integer})
        @argcheck(length(cd) == length(vd) == length(n),
                  DimensionMismatch("the co-moment, the second moment and the count of an EWDownsideBetaState hold one entry per asset, got $(length(cd)), $(length(vd)) and $(length(n))"))
        return new{typeof(cd), typeof(vd), typeof(n)}(cd, vd, n)
    end
end
function EWDownsideBetaState(; cd::AbstractVector{<:Real}, vd::AbstractVector{<:Real},
                             n::AbstractVector{<:Integer})::EWDownsideBetaState
    return EWDownsideBetaState(cd, vd, n)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`EWDownsideBetaState`](@ref), so that the copy shares no vector with the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::EWDownsideBetaState`: A new state, equal to `x`.

# Related

  - [`EWDownsideBetaState`](@ref)
"""
function Base.copy(x::EWDownsideBetaState)
    return EWDownsideBetaState(copy(x.cd), copy(x.vd), copy(x.n))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses to merge two [`EWDownsideBetaState`](@ref): the recursion of the second block starts from the state after the first one, so two states fitted on disjoint blocks do not give the state of their union to the last bit.

# Validation

  - Always throws an `ArgumentError`.

# Related

  - [`EWDownsideBetaState`](@ref)
  - [`partial_fit!`](@ref)
"""
function merge_states(::EWDownsideBetaState, ::EWDownsideBetaState)
    return throw(ArgumentError("an EWDownsideBetaState cannot merge two states fitted on disjoint blocks: the recursion of the second block starts from the state after the first one. Fold the second block into the state of the first with partial_fit!."))
end
"""
    ew_downside_beta_state(::Nothing, X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                           decay::Real)
    ew_downside_beta_state(st::EWDownsideBetaState, X::AbstractMatrix{<:Real},
                           ::AbstractVector{<:Real}, ::Real)
    ew_downside_beta_state(::AbstractPartialFitState, args...)

Returns the state that a fold of [`ew_downside_beta_series!`](@ref) starts from: a new state at zero for no carried state, or a copy of the carried state.

The two moments of a new state take the type of the returns, of the market return and of the decay, as the recursion gives them after its first observation.

# Arguments

  - `st`: The carried state, or `nothing`.
  - `X`: The returns of the step, `observations × assets`.
  - `rm`: The market return of the step, one entry per observation.
  - $(arg_dict[:decay])

# Validation

  - A carried state holds the assets of `X`. A `DimensionMismatch` is thrown otherwise.
  - A carried state is an [`EWDownsideBetaState`](@ref). An `ArgumentError` is thrown otherwise, because the state belongs to another Descriptor.

# Returns

  - `st::EWDownsideBetaState`: The state to fold into.

# Related

  - [`EWDownsideBetaState`](@ref)
  - [`ew_downside_beta_series!`](@ref)
"""
function ew_downside_beta_state(::Nothing, X::AbstractMatrix{<:Real},
                                rm::AbstractVector{<:Real},
                                decay::Real)::EWDownsideBetaState
    Tf = promote_type(eltype(X), eltype(rm), typeof(decay))
    N = size(X, 2)
    return EWDownsideBetaState(zeros(Tf, N), zeros(Tf, N), zeros(Int, N))
end
function ew_downside_beta_state(st::EWDownsideBetaState, X::AbstractMatrix{<:Real},
                                ::AbstractVector{<:Real}, ::Real)::EWDownsideBetaState
    @argcheck(length(st.cd) == size(X, 2),
              DimensionMismatch("the state of this exponentially weighted Descriptor carries $(length(st.cd)) assets, and the step brings $(size(X, 2))"))
    return copy(st)
end
function ew_downside_beta_state(::AbstractPartialFitState, args...)
    return throw(ArgumentError("the state of this EWDownsideBeta is not an EWDownsideBetaState, so it belongs to another Descriptor. A Descriptor folds from its own state, or from no state."))
end
"""
    ew_downside_beta_series!(st::EWDownsideBetaState, X::AbstractMatrix{<:Real},
                             rm::AbstractVector{<:Real}, amsk::AbstractMatrix{Bool},
                             decay::Real, min_obs::Integer, mar::Real) -> Tuple

Run the exponentially weighted lower partial co-moment of the returns against the market from a state, and return the downside betas and the state after them.

This recursion computes the closed form that [`EWDownsideBeta`](@ref) states. Each asset advances its two moments at each of its own valid observations, so the shortfall of the market enters the denominator over the same observations as the co-moment. An observation where the market is above the target adds nothing to either moment, and both decay, so the estimate still moves in a calm market. The batch call [`descriptor`](@ref) runs it from a new state at zero, and a step of the carry fold from the carried state, so a step equals the batch call by construction.

# Algorithm

At each observation:

 1. Compute the shortfall `dm` of the market.
 2. Reset the moments and the count of every inactive asset through [`ew_downside_beta_step!`](@ref), and advance the two moments and the count of every active asset with a finite return.
 3. Write the downside beta `cd / vd` where the count of the asset has reached `min_obs` and `vd` is above zero. Every other cell is `NaN`, and so is a quotient that is not finite.

# Arguments

  - `st`: The state to fold into. Its vectors change in place.
  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - `amsk`: The active mask, `observations × assets`.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - `mar`: The minimum acceptable return.

# Returns

  - `(B, st)::Tuple`: The plain tuple of `B`, the downside betas, `observations × assets`, and `st`, the state after the observations.

# Related

  - [`EWDownsideBeta`](@ref)
  - [`EWDownsideBetaState`](@ref)
  - [`ew_beta_fold`](@ref)
"""
function ew_downside_beta_series!(st::EWDownsideBetaState, X::AbstractMatrix{<:Real},
                                  rm::AbstractVector{<:Real}, amsk::AbstractMatrix{Bool},
                                  decay::Real, min_obs::Integer, mar::Real)
    (; cd, vd, n) = st
    Tf = eltype(cd)
    B = fill(Tf(NaN), size(X))
    om = 1 - decay
    for t in axes(X, 1)
        dm = min(rm[t] - mar, zero(Tf))
        for i in axes(X, 2)
            ew_downside_beta_step!(st, i, X[t, i], amsk[t, i], dm, decay, om, mar)
            b = cd[i] / vd[i]
            if n[i] >= min_obs && vd[i] > zero(Tf) && isfinite(b)
                B[t, i] = b
            end
        end
    end
    return B, st
end
"""
    ew_downside_beta_step!(st::EWDownsideBetaState, i::Integer, x::Real, act::Bool,
                           dm::Real, decay::Real, om::Real, mar::Real) -> nothing

Advances the downside moments of asset `i` at one observation, in place: a reset where the asset is inactive, nothing where its return `x` is missing, and the update of [`EWDownsideBeta`](@ref) otherwise.

# Arguments

  - `st`: The state.
  - `i`: The asset.
  - `x`: The return of the asset.
  - `act`: Whether the asset is active.
  - `dm`: The shortfall of the market.
  - $(arg_dict[:decay])
  - `om`: The weight of a new observation, `1 - decay`.
  - `mar`: The minimum acceptable return.

# Validation

  - Both moments stay finite. Raises an [`IsNonFiniteError`](@ref), because a product of two finite returns can overflow.

# Returns

  - `nothing`.

# Related

  - [`ew_downside_beta_series!`](@ref)
"""
function ew_downside_beta_step!(st::EWDownsideBetaState, i::Integer, x::Real, act::Bool,
                                dm::Real, decay::Real, om::Real, mar::Real)::Nothing
    (; cd, vd, n) = st
    if !act
        cd[i] = zero(eltype(cd))
        vd[i] = zero(eltype(vd))
        n[i] = 0
        return nothing
    end
    if isfinite(x)
        vd[i] = decay * vd[i] + om * (dm * dm)
        cd[i] = decay * cd[i] + om * min(x - mar, zero(x)) * dm
        n[i] += 1
        @argcheck(isfinite(vd[i]) && isfinite(cd[i]),
                  IsNonFiniteError("the downside moments of asset $i are not finite, because the returns overflow the arithmetic. Rescale the returns."))
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted sensitivity of the returns to the falls of the market, at every observation.

A beta that treats a rise and a fall alike does not show which of the two an asset follows. This Descriptor measures the falls alone, through the lower partial co-moment of the asset with the market. It is how far the asset falls when the market falls short of a target. An investor who fears a loss and not a gain uses it in place of the two-sided beta.

Equation 10 of [estrada2002](@cite) defines the downside beta as the cosemivariance of the asset with the market over the semivariance of the market, and Equation 13 estimates it by a regression through the origin of the asset shortfall on the market shortfall. This Descriptor takes the shortfalls below one target `mar` for both, where that work takes each below its own mean, and it weights the observations exponentially. Both are this library's choices.

# Mathematical definition

```math
\\begin{align}
\\beta^{-}_{t,i} &= \\frac{C^{-}_{t,i}}{V^{-}_{t,i}}\\,, \\\\
C^{-}_{t,i} &= \\lambda C^{-}_{t-1,i} + (1 - \\lambda)\\, D_{t,i}\\, D_{m,t}\\,, \\\\
V^{-}_{t,i} &= \\lambda V^{-}_{t-1,i} + (1 - \\lambda)\\, D_{m,t}^2\\,, \\\\
D_{t,i} &= \\min\\left(x_{t,\\,i} - \\mathrm{mar},\\, 0\\right)\\,, \\\\
D_{m,t} &= \\min\\left(r_{m,t} - \\mathrm{mar},\\, 0\\right)\\,.
\\end{align}
```

Where:

  - ``\\beta^{-}_{t,i}``: Downside beta of asset ``i`` after observation ``t``.
  - ``C^{-}_{t,i}``: Exponentially weighted co-moment of the shortfalls of asset ``i`` and of the market.
  - ``V^{-}_{t,i}``: Exponentially weighted second moment of the shortfall of the market, over the observations of asset ``i``.
  - ``D_{t,i}``, ``D_{m,t}``: Shortfalls of asset ``i`` and of the market below the target.
  - ``\\mathrm{mar}``: `mar`, the minimum acceptable return.
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:r_mt_ewb])
  - $(math_dict[:lambda_ew])

Both moments of an asset advance only at its valid observations, where it is active and its return is finite, so the two sums share their observations and their weights cancel in the quotient. An asset that is listed late, or that misses an observation, therefore carries no bias toward zero. Both moments start from zero, and no mean is subtracted, so a co-moment is not a covariance. An inactive observation resets the asset. The downside beta is `NaN` before the asset has `min_obs` valid observations, and where the market has fallen short of the target on none of them.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWDownsideBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                   decay::Real = half_life_decay(half_life),
                   min_obs::Integer = half_life_min_obs(half_life), mar::Real = 0.0,
                   cache::Option{<:AbstractPartialFitState} = nothing) -> EWDownsideBeta

Keywords correspond to the struct's fields, except `half_life`, which is not a field. It sets the defaults of `decay` and `min_obs`, and a value passed for either of those is used as it is. With daily observations, the default half-life of `60` is about one quarter of a year. The default target of zero makes every loss a shortfall.

## Validation

  - $(val_dict[:decay])
  - $(val_dict[:min_obs])
  - `isfinite(mar)`.

# Examples

```jldoctest
julia> EWDownsideBeta(; half_life = 2)
EWDownsideBeta
     mcap ┼ String: "market_cap"
    decay ┼ Float64: 0.7071067811865476
  min_obs ┼ Int64: 2
      mar ┴ Float64: 0.0
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWBeta`](@ref)
  - [`EWMarketBeta`](@ref)
  - [`ew_downside_beta_series!`](@ref)
  - [`market_return_series`](@ref)

# References

  - $(ref_dict[:estrada2002]) Equations 10 and 13.
"""
@concrete struct EWDownsideBeta <: AbstractDescriptorEstimator
    """
    $(field_dict[:mcap])
    """
    mcap
    """
    $(field_dict[:decay])
    """
    decay
    """
    $(field_dict[:min_obs])
    """
    min_obs
    """
    Minimum acceptable return, the target below which the Descriptor measures the shortfalls of the asset and of the market.
    """
    mar
    """
    $(field_dict[:ew_beta_desc_cache])
    """
    cache
    function EWDownsideBeta(mcap::AbstractString, decay::Real, min_obs::Integer, mar::Real,
                            cache::Option{<:AbstractPartialFitState})
        assert_panel_terms(mcap, :mcap)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_finite(mar, :mar)
        return new{typeof(mcap), typeof(decay), typeof(min_obs), typeof(mar),
                   typeof(cache)}(mcap, decay, min_obs, mar, cache)
    end
end
function EWDownsideBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                        decay::Real = half_life_decay(half_life),
                        min_obs::Integer = half_life_min_obs(half_life), mar::Real = 0.0,
                        cache::Option{<:AbstractPartialFitState} = nothing)::EWDownsideBeta
    return EWDownsideBeta(mcap, decay, min_obs, mar, cache)
end
"""
    descriptor(de::EWDownsideBeta, rd::ReturnsResult) -> Matrix{<:Real}

Compute an exponentially weighted downside beta Descriptor from a [`ReturnsResult`](@ref).

# Algorithm

 1. Build the market return `rm` through [`market_return_series`](@ref).
 2. Run the recursion from a new state through [`ew_downside_beta_series!`](@ref), for the Descriptor `D`. [`ew_beta_fold`](@ref) holds these steps, and a step of the carry fold runs them from the carried state.
 3. Write `NaN` into the inactive cells of `D` through [`descriptor_active_fill!`](@ref).

An asset whose return is missing at an observation keeps its two moments there, so its downside beta holds.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - The rules of [`market_return_series`](@ref).
  - The moments stay finite. Raises an [`IsNonFiniteError`](@ref).

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05], pnl = pnl);

julia> descriptor(EWDownsideBeta(; half_life = 1), rd)
3×2 Matrix{Float64}:
 NaN        NaN
   2.33333    0.0
   2.33333    0.0
```

# Related

  - [`EWDownsideBeta`](@ref)
  - [`ew_downside_beta_series!`](@ref)
  - [`market_return_series`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function descriptor(de::EWDownsideBeta, rd::ReturnsResult)::Matrix{<:Real}
    return ew_beta_fold(de, rd, nothing).D
end
"""
    ew_macro_state(::Nothing, de::EWMacroSensitivity, X::AbstractMatrix{<:Real},
                   rm::AbstractVector{<:Real}, rf::AbstractVector{<:Real})
    ew_macro_state(st::EWMacroSensitivityState, de::EWMacroSensitivity,
                   X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                   rf::AbstractVector{<:Real})

Returns the state that a fold of an [`EWMacroSensitivity`](@ref) starts from: a new state for no carried state, with every constant-market flag set, or a copy of the carried state.

# Arguments

  - `st`: The carried state, or `nothing`.
  - `de`: Descriptor Estimator.
  - `X`: The returns of the step, `observations × assets`.
  - `rm`: The market return of the step.
  - `rf`: The reference return of the step.

# Validation

  - The rules of [`ew_beta_state`](@ref) for the beta state that the state holds.

# Returns

  - `st::EWMacroSensitivityState`: The state to fold into.

# Related

  - [`EWMacroSensitivityState`](@ref)
  - [`ew_beta_fold`](@ref)
"""
function ew_macro_state(::Nothing, de::EWMacroSensitivity, X::AbstractMatrix{<:Real},
                        rm::AbstractVector{<:Real},
                        rf::AbstractVector{<:Real})::EWMacroSensitivityState
    be = ew_beta_state(nothing, X, rm, rf, de.decay, nothing, de.agg_obs)
    return EWMacroSensitivityState(be, trues(size(X, 2)), 0, false)
end
function ew_macro_state(st::EWMacroSensitivityState, de::EWMacroSensitivity,
                        X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                        rf::AbstractVector{<:Real})::EWMacroSensitivityState
    be = ew_beta_state(st.be, X, rm, rf, de.decay, nothing, de.agg_obs)
    return EWMacroSensitivityState(be, copy(st.cm), st.c, st.hf)
end
"""
    ew_macro_carried(::Nothing) -> CarriedReference
    ew_macro_carried(st::EWMacroSensitivityState) -> CarriedReference
    ew_macro_carried(::AbstractPartialFitState)

Returns what the refusals of an [`EWMacroSensitivity`](@ref) read off its carried state: the count `c` of the windows with a finite reference return, whether the open window holds one, `hf`, and the number `t0` of the observations that the state folded. The batch call starts from no state, with every value at zero.

# Arguments

  - `st`: The carried state, or `nothing`.

# Validation

  - A carried state is an [`EWMacroSensitivityState`](@ref). An `ArgumentError` is thrown otherwise, because the state belongs to another Descriptor.

# Returns

  - `carried::CarriedReference`: `c`, `hf` and `t0`.

# Related

  - [`ew_macro_windows`](@ref)
  - [`EWMacroSensitivityState`](@ref)
"""
function ew_macro_carried(::Nothing)::CarriedReference
    return CarriedReference(0, false, 0)
end
function ew_macro_carried(st::EWMacroSensitivityState)::CarriedReference
    return CarriedReference(st.c, st.hf, st.be.t)
end
function ew_macro_carried(::AbstractPartialFitState)
    return throw(ArgumentError("the state of this EWMacroSensitivity is not an EWMacroSensitivityState, so it belongs to another Descriptor. A Descriptor folds from its own state, or from no state."))
end
"""
    ew_beta_fold(de::EWBeta, rd::ReturnsResult, cache::Option{<:AbstractPartialFitState})
    ew_beta_fold(de::EWMacroSensitivity, rd::ReturnsResult,
                 cache::Option{<:AbstractPartialFitState};
                 ref::Option{<:AbstractVector{<:Real}} = nothing)
    ew_beta_fold(de::EWDownsideBeta, rd::ReturnsResult,
                 cache::Option{<:AbstractPartialFitState})

Folds the observations of a [`ReturnsResult`](@ref) into the recursion of an exponentially weighted beta Descriptor from a state, and returns the state after them and the Descriptor of each observation.

The batch call [`descriptor`](@ref) is this function from no state, so a step equals the batch call by construction. Every quantity that a recursion reads at an observation is a quantity of that observation, such as its market return, its active mask, its group labels and its capitalisation, or a quantity that the state carries. An [`EWBeta`](@ref) and an [`EWMacroSensitivity`](@ref) with `agg_obs > 1` carry the open window in their [`EWBetaState`](@ref), and count its observations from the first one that the state folded.

# Algorithm

 1. Check the data and build the market return as [`descriptor`](@ref) does. An [`EWMacroSensitivity`](@ref) first selects its reference return through [`ew_macro_reference`](@ref), refuses an infinite value of it, and checks its windows through [`ew_macro_windows`](@ref), from the counts that [`ew_macro_carried`](@ref) reads off the state.
 2. Take the state from [`ew_beta_state`](@ref), [`ew_macro_state`](@ref) or [`ew_downside_beta_state`](@ref).
 3. Run the recursion from the state, through [`ew_regression_series!`](@ref) or [`ew_downside_beta_series!`](@ref).
 4. Write `NaN` where an observation is inactive, through [`descriptor_active_fill!`](@ref).

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `cache`: The carried state, or `nothing` for the first observation.
  - `ref`: The keyword `ref` of [`descriptor`](@ref).

# Validation

  - The rules of [`descriptor`](@ref), of [`ew_beta_state`](@ref), of [`ew_macro_state`](@ref), of [`ew_macro_carried`](@ref) and of [`ew_downside_beta_state`](@ref).

# Returns

  - `fold::DescriptorFold`: `st`, the state after the observations, and `D`, the Descriptor of each observation, `observations × assets`.

# Related

  - [`descriptor_step`](@ref)
  - [`EWBetaState`](@ref)
  - [`EWMacroSensitivityState`](@ref)
  - [`EWDownsideBetaState`](@ref)
"""
function ew_beta_fold(de::EWBeta, rd::ReturnsResult,
                      cache::Option{<:AbstractPartialFitState})
    pnl = descriptor_asset_panel(rd)
    rm = market_return_series(rd, de.mcap)
    st = ew_beta_state(cache, rd.X, rm, nothing, de.decay, de.group, de.agg_obs)
    md = ew_beta_shrinkage(de.group, de, rd)
    B, bst = ew_regression_series!(st, md, rd.X, rm, nothing, pnl.amsk, de.min_obs,
                                   de.agg_obs)
    descriptor_active_fill!(B, pnl)
    return DescriptorFold(bst, B)
end
function ew_beta_fold(de::EWMacroSensitivity, rd::ReturnsResult,
                      cache::Option{<:AbstractPartialFitState};
                      ref::Option{<:AbstractVector{<:Real}} = nothing)
    pnl = descriptor_asset_panel(rd)
    cr = ew_macro_carried(cache)
    rf = ew_macro_reference(de.series, de, rd, ref)
    t = findfirst(isinf, rf)
    @argcheck(isnothing(t),
              IsNonFiniteError("the reference return must be finite or NaN on every observation, and it is infinite at observation $(cr.t0 + something(t, 0)) of the returns data. NaN is a gap that the recursion skips, and an infinite value is not a gap."))
    c, hf = ew_macro_windows(de.series, rf, de, cr)
    rm = market_return_series(rd, de.mcap)
    st = ew_macro_state(cache, de, rd.X, rm, rf)
    B, be = ew_regression_series!(st.be, st.cm, rd.X, rm, rf, pnl.amsk, de.min_obs,
                                  de.agg_obs)
    descriptor_active_fill!(B, pnl)
    return DescriptorFold(EWMacroSensitivityState(be, st.cm, c, hf), B)
end
function ew_beta_fold(de::EWDownsideBeta, rd::ReturnsResult,
                      cache::Option{<:AbstractPartialFitState})
    pnl = descriptor_asset_panel(rd)
    rm = market_return_series(rd, de.mcap)
    st = ew_downside_beta_state(cache, rd.X, rm, de.decay)
    B, dst = ew_downside_beta_series!(st, rd.X, rm, pnl.amsk, de.decay, de.min_obs, de.mar)
    descriptor_active_fill!(B, pnl)
    return DescriptorFold(dst, B)
end
"""
    descriptor_step(de::Union{EWBeta, EWMacroSensitivity, EWDownsideBeta}, rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into the carried state of an exponentially weighted beta Descriptor, and returns the Descriptor of each one.

The Descriptor of an observation equals the one of the batch call [`descriptor`](@ref) over every observation that the state folded and the observations before it, to the last bit, because the state runs the recursion with the same arithmetic from the same first observation, as [`ew_beta_fold`](@ref) states. The step copies the state, so the estimator it gets keeps its state. An [`EWMacroSensitivity`](@ref) reads its reference return from the column of the Exogenous Series that `series` names, because the step takes no keyword `ref`.

# Arguments

  - `de`: The estimator, with or without a state.
  - $(arg_dict[:rd]) It holds the new observations alone.

# Validation

  - The rules of [`ew_beta_fold`](@ref).

# Returns

  - `step::DescriptorStepResult`: `de`, the estimator with the state after the observations in `cache`, and `D`, the Descriptor of each observation, `observations × assets`.

# Related

  - [`EWBetaState`](@ref)
  - [`EWMacroSensitivityState`](@ref)
  - [`EWDownsideBetaState`](@ref)
  - [`partial_fit!`](@ref)
  - [`descriptor_carry`](@ref)
"""
function descriptor_step(de::Union{EWBeta, EWMacroSensitivity, EWDownsideBeta},
                         rd::ReturnsResult)
    (; st, D) = ew_beta_fold(de, rd, de.cache)
    return DescriptorStepResult(rebuild_struct(de, (; cache = st)), D)
end
"""
    partial_fit!(de::Union{EWBeta, EWMacroSensitivity, EWDownsideBeta}, rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into the carried state of an exponentially weighted beta Descriptor, and returns the estimator with the state after them in `cache`. [`descriptor_step`](@ref) states the fold, and also returns the Descriptor of each observation.

# Arguments

  - `de`: The estimator, with no state or with its state.
  - $(arg_dict[:rd]) It holds the new observations alone.

# Validation

  - The rules of [`descriptor_step`](@ref).

# Returns

  - `de::Union{EWBeta, EWMacroSensitivity, EWDownsideBeta}`: The estimator, with its `cache` field set to the state after the observations.

# Related

  - [`descriptor_step`](@ref)
  - [`EWBetaState`](@ref)
  - [`EWMacroSensitivityState`](@ref)
  - [`EWDownsideBetaState`](@ref)
"""
function partial_fit!(de::Union{EWBeta, EWMacroSensitivity, EWDownsideBeta},
                      rd::ReturnsResult)
    return descriptor_step(de, rd).de
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of an [`EWBeta`](@ref), an [`EWMacroSensitivity`](@ref) or an [`EWDownsideBeta`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one, such as a [`CompositeExposure`](@ref). Set `set_show_nothing_fields!(:EWBeta, true)` to render it.

# Arguments

  - `de`: The estimator.

# Returns

  - `fields::Tuple`: The field names to render, every field name but `:cache`.

# Related

  - [`EWBeta`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(de::Union{EWBeta, EWMacroSensitivity, EWDownsideBeta})
    return filter(!=(:cache), fieldnames(typeof(de)))
end

export EWMacroSensitivity, EWDownsideBeta
