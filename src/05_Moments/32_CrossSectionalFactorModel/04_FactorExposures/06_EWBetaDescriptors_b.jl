"""
$(DocStringExtensions.TYPEDEF)

The carried state of the recursion of an [`EWMacroSensitivity`](@ref), after the windows that it folded.

The state holds every quantity that [`ew_macro_sensitivity_series!`](@ref) reads from the window before: the held partial beta, the mean, the two covariances and the valid count of each asset, the means, the variances and the covariance of the market and of the reference series, the count of the windows with a finite reference return, and the count of the windows. An [`EWBlockState`](@ref) holds it, with the observations of a window that is not yet complete.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWMacroSensitivityState(; b::AbstractVector{<:Real}, mu::AbstractVector{<:Real},
                            cam::AbstractVector{<:Real}, caf::AbstractVector{<:Real},
                            n::AbstractVector{<:Integer}, mf::NamedTuple, c::Integer,
                            t::Integer) -> EWMacroSensitivityState

Keywords correspond to the struct's fields.

## Validation

  - `b`, `mu`, `cam`, `caf` and `n` hold the same number of assets. A `DimensionMismatch` is thrown otherwise.
  - `mf` holds the five keys `mu_m`, `mu_f`, `var_m`, `var_f` and `cov_mf`, each a real number. An `ArgumentError` is thrown otherwise.

# Related

  - [`ew_macro_sensitivity_series!`](@ref)
  - [`EWBlockState`](@ref)
  - [`ew_beta_fold`](@ref)
"""
@concrete struct EWMacroSensitivityState <: AbstractPartialFitState
    """
    The last partial beta ``\\beta^{f}_{t,i}`` of each asset, `NaN` before its first one.
    """
    b
    """
    The exponentially weighted mean of each asset.
    """
    mu
    """
    The exponentially weighted covariance ``C^{m}_{t,i}`` of each asset with the market.
    """
    cam
    """
    The exponentially weighted covariance ``C^{f}_{t,i}`` of each asset with the reference series.
    """
    caf
    """
    The count of the finite returns of each asset.
    """
    n
    """
    The moments of the market and of the reference series, `(; mu_m, mu_f, var_m, var_f, cov_mf)`: the exponentially weighted means of both, their variances ``V^{m}_{t}`` and ``V^{f}_{t}``, and their covariance ``C^{mf}_{t}``.
    """
    mf
    """
    The count of the windows whose reference return is finite.
    """
    c
    """
    The count of the windows that the state folded.
    """
    t
    function EWMacroSensitivityState(b::AbstractVector{<:Real}, mu::AbstractVector{<:Real},
                                     cam::AbstractVector{<:Real},
                                     caf::AbstractVector{<:Real},
                                     n::AbstractVector{<:Integer}, mf::NamedTuple,
                                     c::Integer, t::Integer)
        @argcheck(length(b) == length(mu) == length(cam) == length(caf) == length(n),
                  DimensionMismatch("the vectors of an EWMacroSensitivityState hold one entry per asset, got $(map(length, (b, mu, cam, caf, n)))"))
        @argcheck(keys(mf) == (:mu_m, :mu_f, :var_m, :var_f, :cov_mf) &&
                  all(x -> isa(x, Real), mf),
                  ArgumentError("the moments of an EWMacroSensitivityState are the real numbers (; mu_m, mu_f, var_m, var_f, cov_mf), got $(mf)"))
        return new{typeof(b), typeof(mu), typeof(cam), typeof(caf), typeof(n), typeof(mf),
                   typeof(c), typeof(t)}(b, mu, cam, caf, n, mf, c, t)
    end
end
function EWMacroSensitivityState(; b::AbstractVector{<:Real}, mu::AbstractVector{<:Real},
                                 cam::AbstractVector{<:Real}, caf::AbstractVector{<:Real},
                                 n::AbstractVector{<:Integer}, mf::NamedTuple, c::Integer,
                                 t::Integer)::EWMacroSensitivityState
    return EWMacroSensitivityState(b, mu, cam, caf, n, mf, c, t)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies an [`EWMacroSensitivityState`](@ref), so that the copy shares no vector with the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::EWMacroSensitivityState`: A new state, equal to `x`.

# Related

  - [`EWMacroSensitivityState`](@ref)
"""
function Base.copy(x::EWMacroSensitivityState)
    return EWMacroSensitivityState(copy(x.b), copy(x.mu), copy(x.cam), copy(x.caf),
                                   copy(x.n), x.mf, x.c, x.t)
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
$(DocStringExtensions.TYPEDSIGNATURES)

Returns a new state at zero for [`ew_macro_sensitivity_series!`](@ref).

The vectors take the type of the returns, of the market return and of the reference return, and the five scalars of the market and of the reference series also take the type of the decay, as the recursion gives them after its first window. The first window reads the same values from both, so the batch call keeps every bit.

# Arguments

  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - `rf`: The reference return, one entry per observation.
  - $(arg_dict[:decay])

# Returns

  - `st::EWMacroSensitivityState`: The state at zero, with every partial beta `NaN`.

# Related

  - [`EWMacroSensitivityState`](@ref)
  - [`ew_block_state`](@ref)
"""
function ew_macro_sensitivity_state(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                                    rf::AbstractVector{<:Real},
                                    decay::Real)::EWMacroSensitivityState
    Tf = float_if_integer(promote_type(eltype(X), eltype(rm), eltype(rf)))
    N = size(X, 2)
    z = zero(promote_type(Tf, typeof(decay)))
    return EWMacroSensitivityState(fill(Tf(NaN), N), zeros(Tf, N), zeros(Tf, N),
                                   zeros(Tf, N), zeros(Int, N),
                                   (; mu_m = z, mu_f = z, var_m = z, var_f = z, cov_mf = z),
                                   0, 0)
end
"""
    ew_macro_sensitivity_series!(st::EWMacroSensitivityState, X::AbstractMatrix{<:Real},
                                 rm::AbstractVector{<:Real}, rf::AbstractVector{<:Real},
                                 decay::Real, min_obs::Integer, min_val::Real) -> NamedTuple

Run the exponentially weighted partial beta of the returns on a reference series from a state, and return the partial betas and the state after them.

This recursion computes the closed form that [`EWMacroSensitivity`](@ref) states, from the exponentially weighted moments alone and with no matrix to invert. The batch call [`descriptor`](@ref) runs it from a new state at zero, and a step of the carry fold from the carried state, so a step equals the batch call by construction.

# Algorithm

 1. Skip an observation whose reference return is not finite. The whole state holds its value there, the market state included, because a partial beta needs both series at one observation.
 2. Count the observation in `c`.
 3. Take the deviations `dm` and `df` of the market and of the reference series from the means of the previous observation.
 4. Advance the means `mu_m` and `mu_f`, the variances `var_m` and `var_f`, and the covariance `cov_mf`.
 5. For each asset with a finite return, take its deviation `d` from its previous mean.
 6. Advance the mean `mu`, the covariances `cam` and `caf`, and the valid count `n` of that asset. An asset without a finite return keeps its state.
 7. Write a new partial beta where `c` and the valid count of the asset have both reached `min_obs`. An asset keeps its last partial beta at every other observation, and it is `NaN` before its first one.

# Arguments

  - `st`: The state to fold into. Its vectors change in place.
  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - `rf`: The reference return, one entry per observation.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - $(arg_dict[:min_val])

# Returns

  - `fold::NamedTuple`: `B`, the partial betas, `observations × assets`, before any mask is applied, and `st`, the state after the observations.

# Related

  - [`EWMacroSensitivity`](@ref)
  - [`EWMacroSensitivityState`](@ref)
  - [`ew_beta_fold`](@ref)
  - [`ew_beta_series`](@ref)
"""
function ew_macro_sensitivity_series!(st::EWMacroSensitivityState,
                                      X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                                      rf::AbstractVector{<:Real}, decay::Real,
                                      min_obs::Integer, min_val::Real)
    (; b, mu, cam, caf, n) = st
    Tf = eltype(b)
    T, N = size(X)
    B = Matrix{Tf}(undef, T, N)
    (; mu_m, mu_f, var_m, var_f, cov_mf) = st.mf
    c = st.c
    om = one(Tf) - decay
    for t in 1:T
        f = rf[t]
        if !isfinite(f)
            B[t, :] = b
            continue
        end
        c += 1
        r = rm[t]
        dm = r - mu_m
        df = f - mu_f
        mu_m = decay * mu_m + om * r
        mu_f = decay * mu_f + om * f
        var_m = decay * var_m + om * dm * dm
        var_f = decay * var_f + om * df * df
        cov_mf = decay * cov_mf + om * dm * df
        vm = var_m + min_val
        vf = var_f - cov_mf * cov_mf / vm + min_val
        for i in 1:N
            x = X[t, i]
            if isfinite(x)
                d = x - mu[i]
                mu[i] = decay * mu[i] + om * x
                cam[i] = decay * cam[i] + om * d * dm
                caf[i] = decay * caf[i] + om * d * df
                n[i] += 1
                if c >= min_obs && n[i] >= min_obs
                    b[i] = (caf[i] - cam[i] * cov_mf / vm) / vf
                end
            end
        end
        B[t, :] = b
    end
    return (; B = B,
            st = EWMacroSensitivityState(b, mu, cam, caf, n,
                                         (; mu_m, mu_f, var_m, var_f, cov_mf), c, st.t + T))
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted sensitivity of the returns to a reference series, after the market is removed.

The exposure of an asset to an exchange rate, a rate of interest, inflation or a basket of commodities is the part of its move that the market does not explain. The partial beta is the coefficient of the reference series in the regression of the returns on the market and the reference series together, so it contains no exposure that the market explains. The reference series belongs to no asset, so it is not a Panel Field. The field `series` names its column in the Exogenous Series `rd.E` of the returns data, and [`descriptor`](@ref) reads that column. A [`CrossSectionalFactorPrior`](@ref) and a cross-validation fold pass the returns data alone, so inside them the field is the only path. A direct call can pass the series as the keyword `ref` instead.

# Mathematical definition

```math
\\begin{align}
x_{t,\\,i} &= \\alpha_i + \\beta^{m}_i\\, r_{m,t} + \\beta^{f}_i\\, r_{f,t} + \\varepsilon_{t,i}\\,, \\\\
\\beta^{f}_{t,i} &= \\frac{C^{f}_{t,i} - C^{m}_{t,i}\\, C^{mf}_{t} / \\tilde{V}^{m}_{t}}{V^{f}_{t} - \\left(C^{mf}_{t}\\right)^2 / \\tilde{V}^{m}_{t} + \\texttt{min\\_val}}\\,, \\\\
\\tilde{V}^{m}_{t} &= V^{m}_{t} + \\texttt{min\\_val}\\,.
\\end{align}
```

Where:

  - ``r_{f,t}``: Reference return at observation ``t``.
  - ``\\alpha_i``, ``\\beta^{m}_i``, ``\\varepsilon_{t,i}``: Intercept, market beta and residual of asset ``i`` in the regression.
  - ``\\beta^{f}_i``: Partial sensitivity of asset ``i`` to the reference series.
  - ``\\beta^{f}_{t,i}``: The Descriptor, the exponentially weighted estimate of ``\\beta^{f}_i`` after observation ``t``. It is the Frisch-Waugh form of the coefficient in the moments below.
  - ``C^{m}_{t,i}``, ``C^{f}_{t,i}``: Exponentially weighted covariances of asset ``i`` with the market and with the reference series.
  - ``C^{mf}_{t}``: Exponentially weighted covariance of the market with the reference series.
  - ``V^{m}_{t}``, ``V^{f}_{t}``: Exponentially weighted variances of the market and of the reference series.
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:r_mt_ewb])
  - $(math_dict[:min_val_ewb])

Each moment follows the recursion that [`EWBeta`](@ref) states, with the decay `decay`, and every moment holds its value at an observation whose reference return is not finite. A series that `series` names is refused where a window it reads after the warm-up holds no finite value, as [`ew_macro_reference`](@ref) states. An infinite reference return is refused on either path.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWMacroSensitivity(; mcap::AbstractString = "market_cap",
                       series::Option{<:AbstractString} = nothing, half_life::Real = 60.0,
                       decay::Real = half_life_decay(half_life),
                       min_obs::Integer = half_life_min_obs(half_life),
                       agg_obs::Integer = 1, min_val::Real = 1e-12,
                       cache::Option{<:AbstractPartialFitState} = nothing) -> EWMacroSensitivity

Keywords correspond to the struct's fields, except `half_life`, which is not a field. It sets the defaults of `decay` and `min_obs`, and a value passed for either of those is used as it is. With daily observations, the default half-life of `60` is about one quarter of a year. The default `series = nothing` names no column, so a call to [`descriptor`](@ref) must then pass the keyword `ref`.

## Validation

  - If `series` is not `nothing`, `!isempty(series)`.
  - $(val_dict[:decay])
  - $(val_dict[:min_obs])
  - `agg_obs >= 1`.
  - `min_val > 0`.

# Examples

```jldoctest
julia> EWMacroSensitivity(; series = \"EURUSD\", half_life = 2)
EWMacroSensitivity
     mcap ┼ String: "market_cap"
   series ┼ String: "EURUSD"
    decay ┼ Float64: 0.7071067811865476
  min_obs ┼ Int64: 2
  agg_obs ┼ Int64: 1
  min_val ┴ Float64: 1.0e-12
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWBeta`](@ref)
  - [`ew_macro_sensitivity_series!`](@ref)
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
    $(field_dict[:min_val])
    """
    min_val
    """
    $(field_dict[:ew_beta_desc_cache])
    """
    cache
    function EWMacroSensitivity(mcap::AbstractString, series::Option{<:AbstractString},
                                decay::Real, min_obs::Integer, agg_obs::Integer,
                                min_val::Real, cache::Option{<:AbstractPartialFitState})
        assert_panel_terms(mcap, :mcap)
        @argcheck(isnothing(series) || !isempty(series),
                  IsEmptyError("series names a column of the Exogenous Series, so it cannot be the empty string"))
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_ew_agg_obs(agg_obs)
        assert_nonempty_gt0_finite_val(min_val, :min_val)
        return new{typeof(mcap), typeof(series), typeof(decay), typeof(min_obs),
                   typeof(agg_obs), typeof(min_val), typeof(cache)}(mcap, series, decay,
                                                                    min_obs, agg_obs,
                                                                    min_val, cache)
    end
end
function EWMacroSensitivity(; mcap::AbstractString = "market_cap",
                            series::Option{<:AbstractString} = nothing,
                            half_life::Real = 60.0,
                            decay::Real = half_life_decay(half_life),
                            min_obs::Integer = half_life_min_obs(half_life),
                            agg_obs::Integer = 1, min_val::Real = 1e-12,
                            cache::Option{<:AbstractPartialFitState} = nothing)::EWMacroSensitivity
    return EWMacroSensitivity(mcap, series, decay, min_obs, agg_obs, min_val, cache)
end
"""
    ew_macro_reference(series::Nothing, de::EWMacroSensitivity, rd::ReturnsResult,
                       ref::Option{<:AbstractVector{<:Real}},
                       ::NamedTuple) -> AbstractVector{<:Real}
    ew_macro_reference(series::AbstractString, de::EWMacroSensitivity, rd::ReturnsResult,
                       ref::Option{<:AbstractVector{<:Real}},
                       carried::NamedTuple) -> AbstractVector{<:Real}

Select the reference return of an [`EWMacroSensitivity`](@ref), from the keyword `ref` or from the column of the Exogenous Series that the field `series` names.

The field and the keyword are two sources for one series, so a call that gives both is refused. The ingestion pads the Exogenous Series with `NaN` where the series is silent, for example before the first value of a series that starts late. The recursion of [`ew_macro_sensitivity_series!`](@ref) reads one value per window of `agg_obs` observations, the mean of the finite values of the window, and it skips a window that holds no finite value. In the warm-up that is correct. After the warm-up, a skipped window freezes the partial beta of every asset, so this function refuses a named series whose window holds no finite value there. A window with a gap and at least one finite value is read as its mean, so it is accepted. The keyword `ref` keeps the rule of the recursion at every window.

# Algorithm

 1. With `series = nothing`, refuse a `ref` that is `nothing` or whose length is not the number of observations, and return `ref`.
 2. Otherwise, refuse a `ref` that is not `nothing`, an `rd.E` that is `nothing`, and a `series` that `rd.ne` does not hold. Take the column `rf` of `rd.E` that `series` names.
 3. Walk the complete windows of `agg_obs` observations. Count in `c` each window that holds a finite value. The windows before `c` reaches `min_obs` are the warm-up of the recursion. A step of the carry fold starts from the count and the observations that its state carries, through [`ew_macro_carried`](@ref), and an observation is numbered from the first one that the state folded.
 4. Refuse the first window after the warm-up that holds no finite value. Return `rf`.

# Arguments

  - `series`: The field `series` of `de`.
  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd])
  - `ref`: The keyword `ref` of [`descriptor`](@ref).
  - `carried`: The values that [`ew_macro_carried`](@ref) reads off the carried state, all at zero for the batch call.

# Validation

  - With `series = nothing`, `ref` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - With `series = nothing`, `length(ref) == size(rd.X, 1)`. Raises a `DimensionMismatch`.
  - With a `series`, `ref` is `nothing`. Raises an `ArgumentError`.
  - With a `series`, `rd.E` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - With a `series`, `rd.ne` holds `series`. Raises an `ArgumentError` that names the series.
  - With a `series`, every window of `rf` after the warm-up holds a finite value. Raises an [`IsNonFiniteError`](@ref) that names the series and the observations of the window.

# Returns

  - `rf::AbstractVector{<:Real}`: The reference return, one entry per observation.

# Related

  - [`EWMacroSensitivity`](@ref)
  - [`descriptor`](@ref)
  - [`ew_macro_sensitivity_series!`](@ref)
  - [`ReturnsResult`](@ref)
"""
function ew_macro_reference(::Nothing, ::EWMacroSensitivity, rd::ReturnsResult,
                            ref::Option{<:AbstractVector{<:Real}},
                            ::NamedTuple)::AbstractVector{<:Real}
    @argcheck(!isnothing(ref),
              IsNothingError("a macro sensitivity is measured against a reference return series, and the estimator names no column of the Exogenous Series in `series`, and the call passes no keyword `ref`. Set `series` to the name of a column of rd.E, or pass the series as `ref`, one entry per observation."))
    @argcheck(length(ref) == size(rd.X, 1),
              DimensionMismatch("the reference return series carries one entry per observation, got length(ref) = $(length(ref)) and size(rd.X, 1) = $(size(rd.X, 1))"))
    return ref
end
function ew_macro_reference(series::AbstractString, de::EWMacroSensitivity,
                            rd::ReturnsResult, ref::Option{<:AbstractVector{<:Real}},
                            carried::NamedTuple)::AbstractVector{<:Real}
    @argcheck(isnothing(ref),
              ArgumentError("the estimator reads its reference return from the column \"$series\" of the Exogenous Series, and the call also passes the keyword `ref`. Give the series in one place: set `series = nothing` to pass `ref`, or do not pass `ref`."))
    @argcheck(!isnothing(rd.E),
              IsNothingError("the estimator reads its reference return from the column \"$series\" of the Exogenous Series of the returns data, and rd.E is nothing. Give the PricesResult or the ReturnsResult an E with a column named \"$series\"."))
    j = findfirst(==(series), rd.ne)
    @argcheck(!isnothing(j),
              ArgumentError("the estimator reads its reference return from the column \"$series\" of the Exogenous Series, and rd.ne has no column of that name. Got rd.ne => $(rd.ne)"))
    rf = view(rd.E, :, j)
    a = de.agg_obs
    (; c, t0) = carried
    A = isnothing(carried.f) ? rf : vcat(carried.f, rf)
    for k in 1:div(length(A), a)
        w = ((k - 1) * a + 1):(k * a)
        f = any(isfinite, view(A, w))
        if c >= de.min_obs && !f
            at = if isone(a)
                "at observation $(t0 + first(w))"
            else
                "at observations $(t0 + first(w)) to $(t0 + last(w))"
            end
            throw(IsNonFiniteError("the reference return \"$series\" must hold a finite value in every window of $a observation(s) that the estimator reads after its warm-up, and it holds none $at of the returns data. A gap in the warm-up is accepted, because the recursion skips it before it writes a sensitivity. Fill the gap, or pass the series as the keyword `ref`, whose recursion holds its state there."))
        end
        c += f
    end
    return rf
end
"""
    descriptor(de::EWMacroSensitivity, rd::ReturnsResult;
               ref::Option{<:AbstractVector{<:Real}} = nothing) -> Matrix{<:Real}

Compute an exponentially weighted macro sensitivity Descriptor from a [`ReturnsResult`](@ref) and a reference series.

The reference series is the column of the Exogenous Series `rd.E` that the field `series` of `de` names. A direct call can pass the series as the keyword `ref` instead, when `series` is `nothing`.

# Algorithm

 1. Select the reference return `rf` through [`ew_macro_reference`](@ref), and refuse an infinite value of it on any observation.
 2. Mask the returns into `X` through [`ew_active_returns`](@ref).
 3. Build the market return `rm` through [`market_return_series`](@ref).
 4. Where `agg_obs` is greater than one, aggregate `X`, `rm` and `rf` into `Xa`, `rma` and `rfa` through [`ew_agg_series`](@ref) and [`ew_agg_vector`](@ref).
 5. Run the recursion from a new state through [`ew_macro_sensitivity_series!`](@ref), for the partial betas `Ba` of every window.
 6. Spread `Ba` over the observations into the Descriptor `D` through [`ew_beta_expand`](@ref).
 7. Write `NaN` into the inactive cells of `D` through [`descriptor_active_fill!`](@ref).

[`ew_beta_fold`](@ref) holds these steps, and a step of the carry fold runs them from the carried state.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`, and the Exogenous Series in `rd.ne` and `rd.E` when `de.series` names a column.
  - `ref`: The reference return, one entry per observation. It is required when `de.series` is `nothing`, and refused otherwise.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - The rules of [`ew_macro_reference`](@ref).
  - No value of the reference return is infinite. `NaN` is a gap, and an infinite value is not. Raises an [`IsNonFiniteError`](@ref) that names the observation.
  - The rules of [`market_return_series`](@ref).

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05], ne = [\"EURUSD\"],
                          E = reshape([0.02, -0.01, 0.03], :, 1), pnl = pnl);

julia> D = descriptor(EWMacroSensitivity(; series = \"EURUSD\", half_life = 1), rd)
3×2 Matrix{Float64}:
  0.070978    0.141956
 15.2941    -10.5882
  1.96733    -1.21431

julia> D == descriptor(EWMacroSensitivity(; half_life = 1), rd; ref = [0.02, -0.01, 0.03])
true
```

# Related

  - [`EWMacroSensitivity`](@ref)
  - [`ew_macro_reference`](@ref)
  - [`ew_macro_sensitivity_series!`](@ref)
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

The state holds every quantity that [`ew_downside_beta_series!`](@ref) reads from the observation before: the co-moment and the valid count of each asset, the second moment of the shortfall of the market, and the count of the observations. A step of the carry fold of a [`CrossSectionalFactorPrior`](@ref) folds its new observations from it with the arithmetic of the batch call.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWDownsideBetaState(; cd::AbstractVector{<:Real}, n::AbstractVector{<:Integer}, vd::Real,
                        t::Integer) -> EWDownsideBetaState

Keywords correspond to the struct's fields.

## Validation

  - `cd` and `n` hold the same number of assets. A `DimensionMismatch` is thrown otherwise.

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
    The count of the finite returns of each asset.
    """
    n
    """
    The second moment ``V^{-}_{t}`` of the shortfall of the market.
    """
    vd
    """
    The count of the observations that the state folded.
    """
    t
    function EWDownsideBetaState(cd::AbstractVector{<:Real}, n::AbstractVector{<:Integer},
                                 vd::Real, t::Integer)
        @argcheck(length(cd) == length(n),
                  DimensionMismatch("the co-moment and the count of an EWDownsideBetaState hold one entry per asset, got $(length(cd)) and $(length(n))"))
        return new{typeof(cd), typeof(n), typeof(vd), typeof(t)}(cd, n, vd, t)
    end
end
function EWDownsideBetaState(; cd::AbstractVector{<:Real}, n::AbstractVector{<:Integer},
                             vd::Real, t::Integer)::EWDownsideBetaState
    return EWDownsideBetaState(cd, n, vd, t)
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
    return EWDownsideBetaState(copy(x.cd), copy(x.n), x.vd, x.t)
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

Returns the state that a fold of [`ew_downside_beta_series!`](@ref) starts from: a new state at zero for no carried state, or a copy of the carried state.

The co-moment of a new state takes the type of the returns and of the market return, and its second moment also takes the type of the decay, as the recursion gives it after its first observation. The first observation reads the same values from both, so the batch call keeps every bit.

# Arguments

  - `st`: The carried state, or `nothing`.
  - `X`: The returns of the step, `observations × assets`.
  - `rm`: The market return of the step, one entry per observation.
  - $(arg_dict[:decay])

# Validation

  - A carried state holds the assets of `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `st::EWDownsideBetaState`: The state to fold into.

# Related

  - [`EWDownsideBetaState`](@ref)
  - [`ew_downside_beta_series!`](@ref)
"""
function ew_downside_beta_state(::Nothing, X::AbstractMatrix{<:Real},
                                rm::AbstractVector{<:Real},
                                decay::Real)::EWDownsideBetaState
    Tf = float_if_integer(promote_type(eltype(X), eltype(rm)))
    N = size(X, 2)
    return EWDownsideBetaState(zeros(Tf, N), zeros(Int, N),
                               zero(promote_type(Tf, typeof(decay))), 0)
end
function ew_downside_beta_state(st::EWDownsideBetaState, X::AbstractMatrix{<:Real},
                                ::AbstractVector{<:Real}, ::Real)::EWDownsideBetaState
    @argcheck(length(st.cd) == size(X, 2),
              DimensionMismatch("the state of this exponentially weighted Descriptor carries $(length(st.cd)) assets, and the step brings $(size(X, 2))"))
    return copy(st)
end
"""
    ew_downside_beta_series!(st::EWDownsideBetaState, X::AbstractMatrix{<:Real},
                             rm::AbstractVector{<:Real}, decay::Real, min_obs::Integer,
                             mar::Real, min_val::Real) -> NamedTuple

Run the exponentially weighted lower partial co-moment of the returns against the market from a state, and return the downside betas and the state after them.

This recursion computes the closed form that [`EWDownsideBeta`](@ref) states. It advances at every observation, so an observation where the market is above the target adds nothing to a co-moment, and the co-moments decay. The estimate therefore still moves in a calm market. A recursion that advanced only when the market fell would hold its value there. The batch call [`descriptor`](@ref) runs it from a new state at zero, and a step of the carry fold from the carried state, so a step equals the batch call by construction.

# Algorithm

 1. At every observation, compute the shortfall `dm` of the market.
 2. Advance the second moment `vd` of the shortfall of the market.
 3. For each asset with a finite return, advance its co-moment `cd` with its own shortfall and `dm`.
 4. Advance the valid count `n` of that asset. An asset without a finite return keeps its co-moment and its count.
 5. Write the downside beta `cd / (vd + min_val)` where the observation count and the valid count of the asset have both reached `min_obs`. The observation count starts at the first observation that the state folded. Every earlier cell is `NaN`.

# Arguments

  - `st`: The state to fold into. Its vectors change in place.
  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - `mar`: The minimum acceptable return.
  - $(arg_dict[:min_val])

# Returns

  - `fold::NamedTuple`: `B`, the downside betas, `observations × assets`, before any mask is applied, and `st`, the state after the observations.

# Related

  - [`EWDownsideBeta`](@ref)
  - [`EWDownsideBetaState`](@ref)
  - [`ew_beta_fold`](@ref)
"""
function ew_downside_beta_series!(st::EWDownsideBetaState, X::AbstractMatrix{<:Real},
                                  rm::AbstractVector{<:Real}, decay::Real, min_obs::Integer,
                                  mar::Real, min_val::Real)
    (; cd, n) = st
    Tf = eltype(cd)
    T, N = size(X)
    B = fill(Tf(NaN), T, N)
    vd = st.vd
    om = one(Tf) - decay
    z = zero(Tf)
    for t in 1:T
        k = st.t + t
        dm = min(rm[t] - mar, z)
        vd = decay * vd + om * dm * dm
        for i in 1:N
            x = X[t, i]
            if isfinite(x)
                cd[i] = decay * cd[i] + om * min(x - mar, z) * dm
                n[i] += 1
            end
            if k >= min_obs && n[i] >= min_obs
                B[t, i] = cd[i] / (vd + min_val)
            end
        end
    end
    return (; B = B, st = EWDownsideBetaState(cd, n, vd, st.t + T))
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted sensitivity of the returns to the falls of the market, at every observation.

A beta that treats a rise and a fall alike does not show which of the two an asset follows. This Descriptor measures the falls alone, through the lower partial co-moment of the asset with the market. It is how far the asset falls when the market falls short of a target. An investor who fears a loss and not a gain uses it in place of the two-sided beta.

# Mathematical definition

```math
\\begin{align}
\\beta^{-}_{t,i} &= \\frac{C^{-}_{t,i}}{V^{-}_{t} + \\texttt{min\\_val}}\\,, \\\\
C^{-}_{t,i} &= \\lambda C^{-}_{t-1,i} + (1 - \\lambda)\\, D_{t,i}\\, D_{m,t}\\,, \\\\
V^{-}_{t} &= \\lambda V^{-}_{t-1} + (1 - \\lambda)\\, D_{m,t}^2\\,, \\\\
D_{t,i} &= \\min\\left(x_{t,\\,i} - \\mathrm{mar},\\, 0\\right)\\,, \\\\
D_{m,t} &= \\min\\left(r_{m,t} - \\mathrm{mar},\\, 0\\right)\\,.
\\end{align}
```

Where:

  - ``\\beta^{-}_{t,i}``: Downside beta of asset ``i`` after observation ``t``.
  - ``C^{-}_{t,i}``: Exponentially weighted co-moment of the shortfalls of asset ``i`` and of the market. It holds its value at an observation where the return of the asset is not finite.
  - ``V^{-}_{t}``: Exponentially weighted second moment of the shortfall of the market.
  - ``D_{t,i}``, ``D_{m,t}``: Shortfalls of asset ``i`` and of the market below the target.
  - ``\\mathrm{mar}``: `mar`, the minimum acceptable return.
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:r_mt_ewb])
  - $(math_dict[:lambda_ew])
  - $(math_dict[:min_val_ewb])

Both states start from zero. No mean is subtracted, so a co-moment is not a covariance.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWDownsideBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                   decay::Real = half_life_decay(half_life),
                   min_obs::Integer = half_life_min_obs(half_life), mar::Real = 0.0,
                   min_val::Real = 1e-12,
                   cache::Option{<:AbstractPartialFitState} = nothing) -> EWDownsideBeta

Keywords correspond to the struct's fields, except `half_life`, which is not a field. It sets the defaults of `decay` and `min_obs`, and a value passed for either of those is used as it is. With daily observations, the default half-life of `60` is about one quarter of a year. The default target of zero makes every loss a shortfall.

## Validation

  - $(val_dict[:decay])
  - $(val_dict[:min_obs])
  - `isfinite(mar)`.
  - `min_val > 0`.

# Examples

```jldoctest
julia> EWDownsideBeta(; half_life = 2)
EWDownsideBeta
     mcap ┼ String: "market_cap"
    decay ┼ Float64: 0.7071067811865476
  min_obs ┼ Int64: 2
      mar ┼ Float64: 0.0
  min_val ┴ Float64: 1.0e-12
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWBeta`](@ref)
  - [`EWMarketBeta`](@ref)
  - [`ew_downside_beta_series!`](@ref)
  - [`market_return_series`](@ref)

# References

  - $(ref_dict[:angchenxing2006])
  - $(ref_dict[:estrada2002])
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
    $(field_dict[:min_val])
    """
    min_val
    """
    $(field_dict[:ew_beta_desc_cache])
    """
    cache
    function EWDownsideBeta(mcap::AbstractString, decay::Real, min_obs::Integer, mar::Real,
                            min_val::Real, cache::Option{<:AbstractPartialFitState})
        assert_panel_terms(mcap, :mcap)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_finite(mar, :mar)
        assert_nonempty_gt0_finite_val(min_val, :min_val)
        return new{typeof(mcap), typeof(decay), typeof(min_obs), typeof(mar),
                   typeof(min_val), typeof(cache)}(mcap, decay, min_obs, mar, min_val,
                                                   cache)
    end
end
function EWDownsideBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                        decay::Real = half_life_decay(half_life),
                        min_obs::Integer = half_life_min_obs(half_life), mar::Real = 0.0,
                        min_val::Real = 1e-12,
                        cache::Option{<:AbstractPartialFitState} = nothing)::EWDownsideBeta
    return EWDownsideBeta(mcap, decay, min_obs, mar, min_val, cache)
end
"""
    descriptor(de::EWDownsideBeta, rd::ReturnsResult) -> Matrix{<:Real}

Compute an exponentially weighted downside beta Descriptor from a [`ReturnsResult`](@ref).

# Algorithm

 1. Build the market return `rm` through [`market_return_series`](@ref).
 2. Mask the returns through [`ew_active_returns`](@ref).
 3. Run the recursion from a new state through [`ew_downside_beta_series!`](@ref), for the Descriptor `D`. [`ew_beta_fold`](@ref) holds these steps, and a step of the carry fold runs them from the carried state.
 4. Write `NaN` into the inactive cells of `D` through [`descriptor_active_fill!`](@ref).

An asset whose return is missing at an observation keeps its co-moment there. Once the asset is ready, its downside beta still changes at that observation, because the second moment of the market advances.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - The rules of [`market_return_series`](@ref).

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
 0.0      0.0
 2.33333  0.0
 2.33333  0.0
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
    ew_block_state(::Nothing, de::EWBeta, X::AbstractMatrix{<:Real},
                   m::AbstractVector{<:Real}, ::Nothing)
    ew_block_state(::Nothing, de::EWMacroSensitivity, X::AbstractMatrix{<:Real},
                   m::AbstractVector{<:Real}, f::AbstractVector{<:Real})
    ew_block_state(st::EWBlockState, de::Union{EWBeta, EWMacroSensitivity},
                   X::AbstractMatrix{<:Real}, ::AbstractVector{<:Real},
                   f::Option{<:AbstractVector{<:Real}})

Returns the [`EWBlockState`](@ref) that a fold of an [`EWBeta`](@ref) or of an [`EWMacroSensitivity`](@ref) starts from: a new state for no carried state, or a copy of the carried state.

A new state holds the recursion at zero, from [`ew_beta_state`](@ref) or from [`ew_macro_sensitivity_state`](@ref), and no observation. Its element types follow the masked returns, the market return and the reference return of the step, as the recursion of the batch call does.

# Arguments

  - `st`: The carried state, or `nothing`.
  - `de`: Descriptor Estimator.
  - `X`: The masked returns of the step, `observations × assets`.
  - `m`: The market return of the step, one entry per observation.
  - `f`: The reference return of the step for an [`EWMacroSensitivity`](@ref), or `nothing`.

# Validation

  - A carried state holds a reference return exactly when `f` is one. An `ArgumentError` is thrown otherwise, because the state belongs to the other Descriptor.
  - A carried state holds the assets of `X`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `st::EWBlockState`: The state to fold into.

# Related

  - [`EWBlockState`](@ref)
  - [`ew_beta_fold`](@ref)
"""
function ew_block_state(::Nothing, de::EWBeta, X::AbstractMatrix{<:Real},
                        m::AbstractVector{<:Real}, ::Nothing)::EWBlockState
    return EWBlockState(ew_beta_state(nothing, X, m, de.decay), similar(X, 0, size(X, 2)),
                        similar(m, 0), nothing, nothing, nothing)
end
function ew_block_state(::Nothing, de::EWMacroSensitivity, X::AbstractMatrix{<:Real},
                        m::AbstractVector{<:Real}, f::AbstractVector{<:Real})::EWBlockState
    return EWBlockState(ew_macro_sensitivity_state(X, m, f, de.decay),
                        similar(X, 0, size(X, 2)), similar(m, 0), similar(f, 0), nothing,
                        nothing)
end
function ew_block_state(st::EWBlockState, de::Union{EWBeta, EWMacroSensitivity},
                        X::AbstractMatrix{<:Real}, ::AbstractVector{<:Real},
                        f::Option{<:AbstractVector{<:Real}})::EWBlockState
    @argcheck(isnothing(f) == isnothing(st.f),
              ArgumentError("the state of this $(nameof(typeof(de))) holds $(isnothing(st.f) ? "no" : "a") reference return, so it belongs to another Descriptor. A Descriptor folds from its own state, or from no state."))
    @argcheck(size(st.X, 2) == size(X, 2),
              DimensionMismatch("the state of this exponentially weighted Descriptor carries $(size(st.X, 2)) assets, and the step brings $(size(X, 2))"))
    return copy(st)
end
"""
    ew_macro_carried(::Nothing, ::Integer)
    ew_macro_carried(st::EWBlockState, agg_obs::Integer)

Returns what the refusals of an [`EWMacroSensitivity`](@ref) read off its carried state: the count `c` of the windows with a finite reference return, the reference return `f` of the window that is not yet complete, the number `t0` of the observations of the complete windows, and the number `n0` of every observation that the state folded.

[`ew_macro_reference`](@ref) walks the windows from `c` and `f`, and numbers an observation from `t0`, so a step refuses the gap that the batch call refuses, and names the same observations. The batch call starts from no state, with every value at zero.

# Arguments

  - `st`: The carried state, or `nothing`.
  - $(arg_dict[:agg_obs])

# Validation

  - The recursion of a carried state is an [`EWMacroSensitivityState`](@ref). An `ArgumentError` is thrown otherwise, because the state belongs to an [`EWBeta`](@ref).

# Returns

  - `carried::NamedTuple`: `c`, `f`, `t0` and `n0`. `f` is `nothing` for no state.

# Related

  - [`ew_macro_reference`](@ref)
  - [`EWBlockState`](@ref)
"""
function ew_macro_carried(::Nothing, ::Integer)
    return (; c = 0, f = nothing, t0 = 0, n0 = 0)
end
function ew_macro_carried(st::EWBlockState, agg_obs::Integer)
    @argcheck(isa(st.st, EWMacroSensitivityState),
              ArgumentError("the state of this EWMacroSensitivity holds no reference return, so it belongs to another Descriptor. A Descriptor folds from its own state, or from no state."))
    t0 = st.st.t * agg_obs
    return (; c = st.st.c, f = st.f, t0 = t0, n0 = t0 + length(st.f))
end
"""
    ew_beta_fold(de::EWBeta, rd::ReturnsResult, cache::Option{<:EWBlockState})
    ew_beta_fold(de::EWMacroSensitivity, rd::ReturnsResult, cache::Option{<:EWBlockState};
                 ref::Option{<:AbstractVector{<:Real}} = nothing)
    ew_beta_fold(de::EWDownsideBeta, rd::ReturnsResult,
                 cache::Option{<:EWDownsideBetaState})

Folds the observations of a [`ReturnsResult`](@ref) into the recursion of an exponentially weighted beta Descriptor from a state, and returns the state after them and the Descriptor of each observation.

The batch call [`descriptor`](@ref) is this function from no state, so a step equals the batch call by construction. Every quantity that a recursion reads at an observation is a quantity of that observation, such as its market return, its group labels and its capitalisation, or a quantity that the state carries. An [`EWBeta`](@ref) and an [`EWMacroSensitivity`](@ref) with `agg_obs > 1` fold a window only when it is complete, and the [`EWBlockState`](@ref) carries the observations of the window that is not.

# Algorithm

 1. Check the data and build the masked returns and the market return as [`descriptor`](@ref) does. An [`EWMacroSensitivity`](@ref) first selects and checks its reference return through [`ew_macro_reference`](@ref), from the count and the observations that [`ew_macro_carried`](@ref) reads off the state.
 2. Take the state from [`ew_block_state`](@ref) or from [`ew_downside_beta_state`](@ref).
 3. Join the carried observations to the new ones through [`ew_block_rows`](@ref), and aggregate the complete windows.
 4. Run the recursion from the state, through [`ew_beta_series!`](@ref), [`ew_macro_sensitivity_series!`](@ref) or [`ew_downside_beta_series!`](@ref).
 5. Spread the windows over the observations through [`ew_beta_output`](@ref) or [`ew_beta_expand`](@ref), from the beta that the state held.
 6. Write `NaN` where an observation is inactive, through [`descriptor_active_fill!`](@ref).

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `cache`: The carried state, or `nothing` for the first observation.
  - `ref`: The keyword `ref` of [`descriptor`](@ref).

# Validation

  - The rules of [`descriptor`](@ref), of [`ew_block_state`](@ref), of [`ew_macro_carried`](@ref) and of [`ew_downside_beta_state`](@ref).

# Returns

  - `fold::NamedTuple`: `st`, the state after the observations, and `D`, the Descriptor of each observation, `observations × assets`.

# Related

  - [`descriptor_step`](@ref)
  - [`EWBlockState`](@ref)
  - [`EWDownsideBetaState`](@ref)
"""
function ew_beta_fold(de::EWBeta, rd::ReturnsResult, cache::Option{<:EWBlockState})
    pnl = descriptor_asset_panel(rd)
    rm = market_return_series(rd, de.mcap)
    X = ew_active_returns(rd.X, pnl)
    st = ew_block_state(cache, de, X, rm, nothing)
    a = de.agg_obs
    xs = ew_block_rows(st.X, X, a)
    ms = ew_block_rows(st.m, rm, a)
    Xa = isone(a) ? xs.A : ew_agg_series(xs.A, a)
    rma = isone(a) ? ms.A : ew_agg_vector(ms.A, a)
    b0 = copy(st.st.b)
    fold = ew_beta_series!(st.st, Xa, rma, de.decay, de.min_obs, de.min_val, nothing)
    out = ew_beta_output(de.group, de, rd, fold.B, fold.Vm, Xa, rma, b0, st)
    descriptor_active_fill!(out.D, pnl)
    return (; st = EWBlockState(fold.st, xs.P, ms.P, nothing, out.v, out.s), D = out.D)
end
function ew_beta_fold(de::EWMacroSensitivity, rd::ReturnsResult,
                      cache::Option{<:EWBlockState};
                      ref::Option{<:AbstractVector{<:Real}} = nothing)
    pnl = descriptor_asset_panel(rd)
    a = de.agg_obs
    cr = ew_macro_carried(cache, a)
    rf = ew_macro_reference(de.series, de, rd, ref, cr)
    t = findfirst(isinf, rf)
    @argcheck(isnothing(t),
              IsNonFiniteError("the reference return must be finite or NaN on every observation, and it is infinite at observation $(cr.n0 + t) of the returns data. NaN is a gap that the recursion skips, and an infinite value is not a gap."))
    X = ew_active_returns(rd.X, pnl)
    rm = market_return_series(rd, de.mcap)
    st = ew_block_state(cache, de, X, rm, rf)
    xs = ew_block_rows(st.X, X, a)
    ms = ew_block_rows(st.m, rm, a)
    fs = ew_block_rows(st.f, rf, a)
    Xa = isone(a) ? xs.A : ew_agg_series(xs.A, a)
    rma = isone(a) ? ms.A : ew_agg_vector(ms.A, a)
    rfa = isone(a) ? fs.A : ew_agg_vector(fs.A, a)
    b0 = copy(st.st.b)
    fold = ew_macro_sensitivity_series!(st.st, Xa, rma, rfa, de.decay, de.min_obs,
                                        de.min_val)
    D = ew_beta_expand(fold.B, size(X, 1), a, b0, size(st.X, 1))
    descriptor_active_fill!(D, pnl)
    return (; st = EWBlockState(fold.st, xs.P, ms.P, fs.P, nothing, nothing), D = D)
end
function ew_beta_fold(de::EWDownsideBeta, rd::ReturnsResult,
                      cache::Option{<:EWDownsideBetaState})
    pnl = descriptor_asset_panel(rd)
    rm = market_return_series(rd, de.mcap)
    X = ew_active_returns(rd.X, pnl)
    (; B, st) = ew_downside_beta_series!(ew_downside_beta_state(cache, X, rm, de.decay), X,
                                         rm, de.decay, de.min_obs, de.mar, de.min_val)
    descriptor_active_fill!(B, pnl)
    return (; st = st, D = B)
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

  - `step::NamedTuple`: `de`, the estimator with the state after the observations in `cache`, and `D`, the Descriptor of each observation, `observations × assets`.

# Related

  - [`EWBlockState`](@ref)
  - [`EWDownsideBetaState`](@ref)
  - [`partial_fit!`](@ref)
  - [`descriptor_carry`](@ref)
"""
function descriptor_step(de::Union{EWBeta, EWMacroSensitivity, EWDownsideBeta},
                         rd::ReturnsResult)
    (; st, D) = ew_beta_fold(de, rd, de.cache)
    return (; de = Accessors.@set(de.cache = st), D = D)
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
  - [`EWBlockState`](@ref)
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
