"""
    ew_masked_mean(v::AbstractVector{<:Number}, msk::AbstractVector{Bool}) -> Number

Mean of the entries a mask selects.

# Arguments

  - `v`: The values.
  - `msk`: The mask, of the same length as `v`.

# Returns

  - `m::Real`: The mean of `v[msk]`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_masked_mean([1.0, 2.0, 6.0], [true, false, true])
3.5
```

# Related

  - [`ew_masked_weighted_mean`](@ref)
  - [`ew_beta_shrink`](@ref)
"""
function ew_masked_mean(v::AbstractVector{<:Number}, msk::AbstractVector{Bool})::Number
    Tf = eltype(v)
    s = zero(Tf)
    c = 0
    for i in eachindex(v, msk)
        if msk[i]
            s += v[i]
            c += 1
        end
    end
    return s / c
end
"""
    ew_masked_weighted_mean(v::AbstractVector{<:Number}, w::AbstractVector{<:Number},
                            msk::AbstractVector{Bool}) -> Number

Weighted mean of the entries a mask selects.

The shrinkage of [`EWBeta`](@ref) weights the mean beta of a group by capitalisation, so a large asset moves the mean of its group more than a small asset does.

# Arguments

  - `v`: The values.
  - `w`: The weights, of the same length as `v`.
  - `msk`: The mask, of the same length as `v`.

# Returns

  - `m::Real`: The weighted mean of `v[msk]`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_masked_weighted_mean([1.0, 2.0, 6.0], [1.0, 1.0, 3.0],
                                                   [true, false, true])
4.75
```

# Related

  - [`ew_masked_mean`](@ref)
  - [`ew_beta_shrink`](@ref)
"""
function ew_masked_weighted_mean(v::AbstractVector{<:Number}, w::AbstractVector{<:Number},
                                 msk::AbstractVector{Bool})::Number
    Tf = promote_type(eltype(v), eltype(w))
    s = zero(Tf)
    d = zero(Tf)
    for i in eachindex(v, w, msk)
        if msk[i]
            s += w[i] * v[i]
            d += w[i]
        end
    end
    return s / d
end
"""
    ew_beta_group_prior(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                        w::AbstractVector{<:Real},
                        msk::AbstractVector{Bool}) -> Tuple{Real, Real}

Estimate the mean and the prior variance of the betas a mask selects.

The two values are ``\\mu_{t,g}`` and ``\\tau^2_{t,g}`` of the definition that [`ew_beta_shrink`](@ref) states. The prior variance is the observed dispersion of the betas less the mean estimation error variance of the group. It is floored at zero, because a group whose estimation noise exceeds its observed dispersion shows no dispersion of its own.

# Arguments

  - `b`: The raw betas.
  - `bev`: The estimation error variance of each beta.
  - `w`: The capitalisation weights.
  - `msk`: The mask that selects the group.

# Returns

  - `m::Real`: The capitalisation-weighted mean beta of the group, ``\\mu_{t,g}``.
  - `pv::Real`: The prior variance of the group, ``\\tau^2_{t,g}``.

# Related

  - [`ew_beta_shrink`](@ref)
  - [`EWBeta`](@ref)
"""
function ew_beta_group_prior(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                             w::AbstractVector{<:Real},
                             msk::AbstractVector{Bool})::Tuple{Real, Real}
    m = ew_masked_weighted_mean(b, w, msk)
    v = ew_masked_weighted_mean((b .- m) .^ 2, w, msk)
    return m, max(v - ew_masked_mean(bev, msk), zero(v))
end
"""
    ew_beta_shrink(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                   L::AbstractVector{<:Integer}, w::AbstractVector{<:Real},
                   min_group_size::Integer, bounds::Tuple{Real, Real}) -> Vector{<:Real}

Shrink one cross-section of raw betas toward the capitalisation-weighted mean of its group.

A beta estimated from few observations is mostly noise, and the mean beta of the assets in its industry is then a better estimate of it. The empirical Bayes weight is the share of the dispersion of the group that is not noise, so a precise beta keeps most of its own value and a noisy beta moves most of the way to the mean of its group.

# Mathematical definition

```math
\\begin{align}
\\beta^{\\ast}_{t,i} &= q_{t,i}\\, \\beta_{t,i} + (1 - q_{t,i})\\, \\mu_{t,g}\\,, \\\\
q_{t,i} &= \\mathrm{clamp}\\!\\left(\\frac{\\tau^2_{t,g}}{\\tau^2_{t,g} + \\sigma^2_{t,i}},\\, q_{\\mathrm{lo}},\\, q_{\\mathrm{hi}}\\right)\\,, \\\\
\\mu_{t,g} &= \\frac{\\sum_{j \\in g} c_{t,j}\\, \\beta_{t,j}}{\\sum_{j \\in g} c_{t,j}}\\,, \\\\
\\tau^2_{t,g} &= \\max\\!\\left(\\frac{\\sum_{j \\in g} c_{t,j} \\left(\\beta_{t,j} - \\mu_{t,g}\\right)^2}{\\sum_{j \\in g} c_{t,j}} - \\frac{1}{\\lvert g \\rvert} \\sum_{j \\in g} \\sigma^2_{t,j},\\, 0\\right)\\,.
\\end{align}
```

Where:

  - ``\\beta^{\\ast}_{t,i}``: Shrunk beta of asset ``i``.
  - ``q_{t,i}``: Weight that the shrunk beta keeps on the raw beta. It is ``q_{\\mathrm{lo}}`` where ``\\tau^2_{t,g}`` is zero.
  - ``g``: Group of asset ``i``, the assets of the estimation set that carry its label. A group smaller than `min_group_size` takes the whole estimation set in its place.
  - ``\\mu_{t,g}``: Capitalisation-weighted mean beta of group ``g``.
  - ``\\tau^2_{t,g}``: Prior variance of group ``g``, its observed dispersion less its mean estimation error variance.
  - ``\\sigma^2_{t,i}``: Estimation error variance of ``\\beta_{t,i}``, from [`ew_beta_variance`](@ref).
  - ``c_{t,j}``: Capitalisation of asset ``j`` at observation ``t``.
  - ``q_{\\mathrm{lo}}``, ``q_{\\mathrm{hi}}``: The two entries of `bounds`.
  - $(math_dict[:beta_ti_ewb])

# Algorithm

 1. Build the shrinkable set, the assets whose beta is finite, whose group label is set, and whose weight is finite and strictly positive. An asset outside it keeps its raw beta.
 2. Build the estimation set `vld`, the shrinkable assets whose estimation error variance is finite. A shrinkable asset outside `vld` has no measure of its own noise, so its beta turns `NaN`. An empty `vld` returns the betas there.
 3. Compute the mean `gm` and the prior variance `gpv` of the whole estimation set through [`ew_beta_group_prior`](@ref).
 4. For each group label in `vld`, compute its mean `m` and its prior variance `pv` through [`ew_beta_group_prior`](@ref). A group with fewer than `min_group_size` members takes `gm` and `gpv` instead.
 5. For each member of the group, compute the weight `q`, zero where `pv` is zero, and clamp it to `bounds`.
 6. Write the shrunk beta, `q` times the raw beta plus `1 - q` times `m`.

# Arguments

  - `b`: The raw betas.
  - `bev`: The estimation error variance of each beta, ``\\sigma^2_{t,i}``.
  - `L`: The group label of each asset, [`CS_MISSING_GROUP`](@ref) where the asset carries none.
  - `w`: The capitalisation weights.
  - $(arg_dict[:min_group_size_ewb])
  - $(arg_dict[:bounds_ewb])

# Returns

  - `s::Vector{<:Real}`: The shrunk betas.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_beta_shrink([0.8, 1.4], [0.01, 0.01], [1, 1], [1.0, 1.0], 1,
                                          (0.0, 1.0))
2-element Vector{Float64}:
 0.8333333333333334
 1.3666666666666667
```

# Related

  - [`EWBeta`](@ref)
  - [`ew_beta_group_prior`](@ref)
  - [`ew_beta_variance`](@ref)
  - [`cross_sectional_groups`](@ref)
  - [`CS_MISSING_GROUP`](@ref)
"""
function ew_beta_shrink(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                        L::AbstractVector{<:Integer}, w::AbstractVector{<:Real},
                        min_group_size::Integer, bounds::Tuple{Real, Real})
    s = collect(b)
    sh = [isfinite(b[i]) &&
              L[i] != CS_MISSING_GROUP &&
              isfinite(w[i]) &&
              w[i] > zero(eltype(w)) for i in eachindex(b, L, w)]
    vld = sh .& isfinite.(bev)
    s[sh .& .!vld] .= eltype(s)(NaN)
    if !any(vld)
        return s
    end
    gm, gpv = ew_beta_group_prior(b, bev, w, vld)
    msk = similar(vld)
    for g in unique(L[i] for i in eachindex(L) if vld[i])
        msk .= vld .& (L .== g)
        m, pv = if count(msk) < min_group_size
            (gm, gpv)
        else
            ew_beta_group_prior(b, bev, w, msk)
        end
        ew_beta_shrink_group!(s, b, bev, msk, m, pv, bounds)
    end
    return s
end
"""
    ew_beta_shrink_group!(s::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                          bev::AbstractVector{<:Real}, msk::AbstractVector{Bool}, m::Real,
                          pv::Real, bounds::Tuple{Real, Real}) -> nothing

Writes the shrunk beta of every member of one group into `s`, in place, as [`ew_beta_shrink`](@ref) states.

# Arguments

  - `s`: The shrunk betas, changed in place.
  - `b`: The raw betas.
  - `bev`: The estimation error variance of each beta.
  - `msk`: The mask of the members of the group.
  - `m`: The mean of the group.
  - `pv`: The prior variance of the group.
  - $(arg_dict[:bounds_ewb])

# Returns

  - `nothing`.

# Related

  - [`ew_beta_shrink`](@ref)
"""
function ew_beta_shrink_group!(s::AbstractVector{<:Real}, b::AbstractVector{<:Real},
                               bev::AbstractVector{<:Real}, msk::AbstractVector{Bool},
                               m::Real, pv::Real, bounds::Tuple{Real, Real})::Nothing
    lo, hi = bounds
    for i in eachindex(msk)
        if msk[i]
            q = clamp(pv > zero(pv) ? pv / (pv + bev[i]) : zero(pv), lo, hi)
            s[i] = q * b[i] + (one(q) - q) * m
        end
    end
    return nothing
end
"""
    ew_beta_variance(mo::EWMoments, sq::EWMoments, b::Real, i::Integer) -> Real

Estimate the sampling variance of the raw market beta `b` of asset `i`, the noise that the shrinkage of [`EWBeta`](@ref) weighs each beta against.

The regression has an intercept and a slope, and its errors are taken as independent with one variance. The exponential weights enter twice: the weighted residual sum of squares estimates the variance of the errors, and the moments of the market under squared weights measure the leverage of the weighted design. The degrees of freedom that the intercept and the slope take are removed from the residual weight.

# Mathematical definition

```math
\\begin{align}
\\sigma^2_{t,i} &= \\frac{S^{xx}_{t,i} - \\beta_{t,i} S^{mx}_{t,i}}{W_{t,i}\\, \\rho_{t,i}}\\, \\frac{L_{t,i}}{\\left(S^{mm}_{t,i}\\right)^2}\\,, \\\\
L_{t,i} &= Q^{mm}_{t,i} + Q_{t,i} \\left(\\mu^{m,Q}_{t,i} - \\mu^{m}_{t,i}\\right)^2\\,, \\\\
\\rho_{t,i} &= 1 - \\frac{Q_{t,i}}{W_{t,i}^2} - \\frac{L_{t,i}}{S^{mm}_{t,i}\\, W_{t,i}}\\,.
\\end{align}
```

Where:

  - ``\\sigma^2_{t,i}``: Estimation error variance of ``\\beta_{t,i}``.
  - $(math_dict[:beta_ti_ewb])
  - ``S^{xx}_{t,i}``, ``S^{mx}_{t,i}``, ``S^{mm}_{t,i}``: The covariance numerators of the asset with itself, of the market with the asset, and of the market with itself, from the moments `mo`.
  - ``W_{t,i}``: The weight sum of `mo`.
  - ``Q^{mm}_{t,i}``, ``Q_{t,i}``, ``\\mu^{m,Q}_{t,i}``: The covariance numerator of the market, the weight sum and the mean of the market under the squared weights, from `sq`.
  - ``\\mu^{m}_{t,i}``: The mean of the market under the weights of `mo`.
  - ``L_{t,i}``: The leverage of the weighted design, recentred at the mean of `mo`.
  - ``\\rho_{t,i}``: The share of the weight left to the residual after the intercept and the slope.

A residual sum of squares below zero within `64 eps` of its scale is rounding, and it is read as zero. The variance is `NaN` where the beta is not finite, before the asset has three observations, where ``\\rho_{t,i}`` is not above `64 eps`, where the residual sum of squares is still negative, and where the result is not a finite number at or above zero.

# Arguments

  - `mo`: The moments of the regression.
  - `sq`: The moments of the market under squared weights.
  - `b`: The raw beta of the asset.
  - `i`: The asset.

# Returns

  - `v::Real`: The estimation error variance, or `NaN`.

# Related

  - [`ew_beta_shrink`](@ref)
  - [`EWMoments`](@ref)
  - [`EWBetaState`](@ref)
"""
function ew_beta_variance(mo::EWMoments, sq::EWMoments, b::Real, i::Integer)::Real
    K = size(mo.C, 1)
    smm = mo.C[1, 1, i]
    sxx = mo.C[K, K, i]
    W = mo.w[i]
    Q = sq.w[i]
    rl = (sq.C[1, 1, i] + Q * (sq.o[1, i] - mo.o[1, i])^2) / smm
    rho = 1 - Q / W^2 - rl / W
    ess = b * mo.C[1, K, i]
    rss = sxx - ess
    tol = 64 * eps(typeof(rss))
    rss = ifelse(-tol * (sxx + abs(ess)) <= rss < zero(rss), zero(rss), rss)
    v = rss / (W * rho) * rl / smm
    ok = isfinite(b) && mo.n[i] >= 3 && rho > tol && rss >= zero(rss) && isfinite(v)
    return ok && v >= zero(v) ? v : oftype(v, NaN)
end
"""
$(DocStringExtensions.TYPEDEF)

The model of [`ew_regression_series!`](@ref) for an [`EWBeta`](@ref) with a `group`: the group labels and the capitalisation of every observation, and the two settings of the shrinkage.

This type is an implementation detail and is not intended for direct use. [`ew_beta_shrinkage`](@ref) makes it, and [`ew_beta_shrink_row!`](@ref) reads it at each complete window.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`ew_beta_shrinkage`](@ref)
  - [`ew_beta_shrink`](@ref)
"""
@concrete struct EWBetaShrinkage
    """
    `L`: The group label of each asset at each observation, `observations × assets`.
    """
    L
    """
    `W`: The capitalisation of each asset at each observation, `observations × assets`.
    """
    W
    """
    $(field_dict[:min_group_size_ewb])
    """
    min_group_size
    """
    $(field_dict[:bounds_ewb])
    """
    bounds
end
"""
    ew_beta_shrink_row!(sh::EWBetaShrinkage, st::EWBetaState, t::Integer) -> nothing

Shrinks the raw betas of the window that closes at observation `t`, in place, through [`ew_beta_shrink`](@ref), with the group labels and the capitalisation of `t`. The estimation error variance of each beta comes from [`ew_beta_variance`](@ref). The shrunk betas hold until the next window closes, and the next window starts again from the raw betas of the moments.

# Arguments

  - `sh`: The shrinkage.
  - `st`: The state, whose betas change in place.
  - `t`: The observation that closes the window.

# Returns

  - `nothing`.

# Related

  - [`ew_regression_series!`](@ref)
  - [`EWBetaShrinkage`](@ref)
"""
function ew_beta_shrink_row!(sh::EWBetaShrinkage, st::EWBetaState, t::Integer)::Nothing
    b = st.b
    bev = [ew_beta_variance(st.mo, st.sq, b[i], i) for i in eachindex(b)]
    b .= ew_beta_shrink(b, bev, view(sh.L, t, :), view(sh.W, t, :), sh.min_group_size,
                        sh.bounds)
    return nothing
end
"""
    assert_ew_shrinkage_bounds(bounds::Tuple{Real, Real}) -> nothing

Check the bounds an exponentially weighted shrinkage clamps its weight to.

# Arguments

  - `bounds`: The `(lo, hi)` bounds.

# Validation

  - `0 <= lo <= hi <= 1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`EWBeta`](@ref)
  - [`ew_beta_shrink`](@ref)
"""
function assert_ew_shrinkage_bounds(bounds::Tuple{Real, Real})::Nothing
    lo, hi = bounds
    @argcheck(zero(lo) <= lo <= hi <= one(hi),
              DomainError(bounds,
                          "bounds clamp the weight a shrunk beta keeps on its raw value, so they must satisfy `0 <= lo <= hi <= 1`, got $bounds"))
    return nothing
end
"""
    assert_ew_agg_obs(agg_obs::Integer) -> nothing

Check the number of observations an exponentially weighted recursion aggregates into one update.

# Arguments

  - $(arg_dict[:agg_obs])

# Validation

  - `agg_obs >= 1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`EWBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
  - [`EWReturnWindow`](@ref)
"""
function assert_ew_agg_obs(agg_obs::Integer)::Nothing
    @argcheck(agg_obs >= one(agg_obs),
              DomainError(agg_obs,
                          "agg_obs is the number of consecutive observations aggregated into one update of the recursion, so it must be at least one, got $agg_obs"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted beta of the returns against the market return, at every observation.

This is the archetype of every market beta Descriptor. The beta of an asset is the slope of the exponentially weighted regression of its return on the market return, with an intercept. [sharpe1964](@citet) defines the beta of an asset as the slope of the regression of its return on the return of the market, which is the covariance over the variance (pages 438 and 439). The exponential weights and the recursion below are this library's own, and no source for them was found. [`market_return_series`](@ref) builds the market return from the Asset Panel. Where the estimator names a categorical Panel Field in `group`, [`ew_beta_shrink`](@ref) shrinks each raw beta toward the capitalisation-weighted mean beta of its group.

# Mathematical definition

```math
\\begin{align}
\\beta_{t,i} &= \\frac{S^{mx}_{t,i}}{S^{mm}_{t,i}}\\,, \\\\
S^{mx}_{t,i} &= \\sum_{s \\in \\mathcal{V}_{t,i}} w_{s,i} \\left(r_{m,s} - \\mu^{m}_{t,i}\\right)\\left(x_{s,\\,i} - \\mu^{x}_{t,i}\\right)\\,, \\\\
S^{mm}_{t,i} &= \\sum_{s \\in \\mathcal{V}_{t,i}} w_{s,i} \\left(r_{m,s} - \\mu^{m}_{t,i}\\right)^2\\,.
\\end{align}
```

Where:

  - $(math_dict[:beta_ti_ewb])
  - ``\\mathcal{V}_{t,i}``: The valid observations of asset ``i`` up to ``t`` since its last reset: the asset is active there, and its return and the market return are both finite.
  - ``w_{s,i}``: The weight of observation ``s`` for asset ``i``, ``(1 - \\lambda) \\lambda^{k}``, where ``k`` counts the valid observations of the asset after ``s``.
  - ``\\mu^{m}_{t,i}``, ``\\mu^{x}_{t,i}``: The weighted means of the market return and of the return of asset ``i`` over ``\\mathcal{V}_{t,i}``.
  - $(math_dict[:x_ti_ret])
  - $(math_dict[:r_mt_ewb])
  - $(math_dict[:lambda_ew])

Both sums run over the observations of the asset alone, with the same weights, so the beta is the slope of a weighted regression with an intercept. An asset that is listed late, or that misses an observation, weights the market over its own observations, and its beta carries no bias toward zero. [`EWMoments`](@ref) holds the sums, and [`ew_regression_series!`](@ref) runs the recursion. A missing return holds the beta of its asset. An inactive observation resets the asset, so its beta is `NaN` until it has `max(2, min_obs)` valid observations again, as [`ew_model_beta`](@ref) states. The beta is also `NaN` where the market does not vary over the observations of the asset.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWBeta(; mcap::AbstractString = "market_cap", decay::Real, min_obs::Integer,
           agg_obs::Integer = 1, group::Option{<:AbstractString} = nothing,
           min_group_size::Integer = 5, bounds::Tuple{Real, Real} = (0.0, 1.0),
           cache::Option{<:AbstractPartialFitState} = nothing) -> EWBeta

Keywords correspond to the struct's fields. `decay` and `min_obs` take no default, because their values depend on the data frequency. [`EWMarketBeta`](@ref) takes a half-life instead, and converts it through [`half_life_decay`](@ref) and [`half_life_min_obs`](@ref).

## Validation

  - $(val_dict[:decay])
  - $(val_dict[:min_obs])
  - `agg_obs >= 1`.
  - `min_group_size >= 1`.
  - `0 <= bounds[1] <= bounds[2] <= 1`.

# Examples

```jldoctest
julia> EWBeta(; decay = 0.5, min_obs = 2)
EWBeta
            mcap ┼ String: "market_cap"
           decay ┼ Float64: 0.5
         min_obs ┼ Int64: 2
         agg_obs ┼ Int64: 1
           group ┼ nothing
  min_group_size ┼ Int64: 5
          bounds ┴ Tuple{Float64, Float64}: (0.0, 1.0)
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWMarketBeta`](@ref)
  - [`EWDownsideBeta`](@ref)
  - [`ew_regression_series!`](@ref)
  - [`ew_beta_shrink`](@ref)
  - [`market_return_series`](@ref)

# References

  - $(ref_dict[:sharpe1964]) Pages 438 and 439.
"""
@concrete struct EWBeta <: AbstractDescriptorEstimator
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
    $(field_dict[:agg_obs])
    """
    agg_obs
    """
    $(field_dict[:group_ewb])
    """
    group
    """
    $(field_dict[:min_group_size_ewb])
    """
    min_group_size
    """
    $(field_dict[:bounds_ewb])
    """
    bounds
    """
    $(field_dict[:ew_beta_desc_cache])
    """
    cache
    function EWBeta(mcap::AbstractString, decay::Real, min_obs::Integer, agg_obs::Integer,
                    group::Option{<:AbstractString}, min_group_size::Integer,
                    bounds::Tuple{Real, Real}, cache::Option{<:AbstractPartialFitState})
        assert_panel_terms(mcap, :mcap)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_ew_agg_obs(agg_obs)
        assert_nonempty_gt0_finite_val(min_group_size, :min_group_size)
        assert_ew_shrinkage_bounds(bounds)
        return new{typeof(mcap), typeof(decay), typeof(min_obs), typeof(agg_obs),
                   typeof(group), typeof(min_group_size), typeof(bounds), typeof(cache)}(mcap,
                                                                                         decay,
                                                                                         min_obs,
                                                                                         agg_obs,
                                                                                         group,
                                                                                         min_group_size,
                                                                                         bounds,
                                                                                         cache)
    end
end
function EWBeta(; mcap::AbstractString = "market_cap", decay::Real, min_obs::Integer,
                agg_obs::Integer = 1, group::Option{<:AbstractString} = nothing,
                min_group_size::Integer = 5, bounds::Tuple{Real, Real} = (0.0, 1.0),
                cache::Option{<:AbstractPartialFitState} = nothing)::EWBeta
    return EWBeta(mcap, decay, min_obs, agg_obs, group, min_group_size, bounds, cache)
end
"""
    ew_beta_shrinkage(::Nothing, ::EWBeta, ::ReturnsResult) -> nothing
    ew_beta_shrinkage(group::AbstractString, de::EWBeta, rd::ReturnsResult) -> EWBetaShrinkage

Returns the model of [`ew_regression_series!`](@ref) that the `group` slot of an [`EWBeta`](@ref) selects: `nothing` for the raw market beta, or the [`EWBetaShrinkage`](@ref) of the group labels and of the capitalisation of `rd`.

# Arguments

  - `group`: The `group` slot of the estimator.
  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Returns

  - `md::Option{<:EWBetaShrinkage}`: The model.

# Related

  - [`EWBetaShrinkage`](@ref)
  - [`ew_beta_fold`](@ref)
  - [`cross_sectional_groups`](@ref)
"""
function ew_beta_shrinkage(::Nothing, ::EWBeta, ::ReturnsResult)::Nothing
    return nothing
end
function ew_beta_shrinkage(group::AbstractString, de::EWBeta,
                           rd::ReturnsResult)::EWBetaShrinkage
    return EWBetaShrinkage(cross_sectional_groups(descriptor_asset_panel(rd), group),
                           Matrix(descriptor_field_values(rd, de.mcap)), de.min_group_size,
                           de.bounds)
end
"""
    descriptor(de::EWBeta, rd::ReturnsResult) -> Matrix{<:Real}

Compute an exponentially weighted market beta Descriptor from a [`ReturnsResult`](@ref).

# Algorithm

 1. Build the market return `rm` through [`market_return_series`](@ref).
 2. Build a new state through [`ew_beta_state`](@ref), and the shrinkage of the `group` slot through [`ew_beta_shrinkage`](@ref).
 3. Run the regression through [`ew_regression_series!`](@ref), which resets an asset that turns inactive, averages each window where `agg_obs > 1`, and shrinks each complete window where the estimator names a `group`.
 4. Write `NaN` into the inactive cells through [`descriptor_active_fill!`](@ref).

[`ew_beta_fold`](@ref) holds these steps, and a step of the carry fold runs them from the carried state.

An asset whose return is missing at an observation keeps its raw beta. With a group, the shrunk beta of that asset can still change, because the mean and the prior variance of its group change.

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

julia> descriptor(EWMarketBeta(; half_life = 1), rd)
3×2 Matrix{Float64}:
 NaN         NaN
   0.954545    0.954545
   1.03478     0.904026
```

# Related

  - [`EWBeta`](@ref)
  - [`EWMarketBeta`](@ref)
  - [`ew_regression_series!`](@ref)
  - [`ew_beta_fold`](@ref)
  - [`market_return_series`](@ref)
"""
function descriptor(de::EWBeta, rd::ReturnsResult)::Matrix{<:Real}
    return ew_beta_fold(de, rd, nothing).D
end
"""
    EWMarketBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                 decay::Real = half_life_decay(half_life),
                 min_obs::Integer = half_life_min_obs(half_life), agg_obs::Integer = 1,
                 group::Option{<:AbstractString} = nothing, min_group_size::Integer = 5,
                 bounds::Tuple{Real, Real} = (0.0, 1.0)) -> EWBeta

Exponentially weighted sensitivity of an asset to the market portfolio.

The value is the beta of [`EWBeta`](@ref) against the capitalisation-weighted market return, the systematic risk of the capital asset pricing model. An asset of beta two moves on average twice as far as the market. With daily observations, the default half-life of `60` is about one quarter of a year.

# Arguments

  - $(arg_dict[:mcap])
  - `half_life`: Half-life of the recursion, in observations. It sets the defaults of `decay` and `min_obs`.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - $(arg_dict[:agg_obs])
  - $(arg_dict[:group_ewb])
  - $(arg_dict[:min_group_size_ewb])
  - $(arg_dict[:bounds_ewb])

# Returns

  - `de::EWBeta`: The estimator, with the half-life fixed.

# Examples

```jldoctest
julia> EWMarketBeta(; half_life = 2)
EWBeta
            mcap ┼ String: "market_cap"
           decay ┼ Float64: 0.7071067811865476
         min_obs ┼ Int64: 2
         agg_obs ┼ Int64: 1
           group ┼ nothing
  min_group_size ┼ Int64: 5
          bounds ┴ Tuple{Float64, Float64}: (0.0, 1.0)
```

# Related

  - [`EWBeta`](@ref)
  - [`descriptor`](@ref)
  - [`EWDownsideBeta`](@ref)
  - [`half_life_decay`](@ref)
"""
function EWMarketBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                      decay::Real = half_life_decay(half_life),
                      min_obs::Integer = half_life_min_obs(half_life), agg_obs::Integer = 1,
                      group::Option{<:AbstractString} = nothing,
                      min_group_size::Integer = 5,
                      bounds::Tuple{Real, Real} = (0.0, 1.0))::EWBeta
    return EWBeta(; mcap = mcap, decay = decay, min_obs = min_obs, agg_obs = agg_obs,
                  group = group, min_group_size = min_group_size, bounds = bounds)
end

export EWBeta, EWMarketBeta
