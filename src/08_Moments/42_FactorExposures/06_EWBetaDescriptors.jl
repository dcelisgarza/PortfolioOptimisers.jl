"""
    ew_active_returns(X::AbstractMatrix{<:Real}, pnl::AssetPanel) -> Matrix{<:Real}

Copy the returns with every inactive cell written to `NaN`.

An asset outside the universe has no return at all, so the cell must not advance a recursion. Every market-relative Descriptor takes this copy before it starts, and its state is then frozen wherever the asset is not listed rather than advanced by a number the panel does not stand behind.

# Arguments

  - `X`: The returns, `observations × assets`.
  - `pnl`: The Asset Panel whose active mask is read.

# Returns

  - `Xm::Matrix{<:Real}`: The returns, `NaN` wherever the active mask is `false`.

# Related

  - [`EWBeta`](@ref)
  - [`EWDownsideBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function ew_active_returns(X::AbstractMatrix{<:Real}, pnl::AssetPanel)
    Xm = Matrix{float(eltype(X))}(X)
    descriptor_active_fill!(Xm, pnl)
    return Xm
end
"""
    ew_agg_series(A::AbstractMatrix{<:Real}, agg_obs::Integer) -> Matrix{<:Real}

Aggregate a matrix into one row per complete window of consecutive observations.

An exponentially weighted beta of two series that do not trade on the same clock reads a covariance that the difference between the clocks pushes toward zero. A window of several observations carries both moves, so the aggregated series measures the covariance the raw series hides. Each cell is the mean of the finite entries of its window, and a window whose entries are all missing is `NaN`. A tail shorter than one window is dropped, so an aggregated series never mixes a complete window with a partial one.

# Arguments

  - `A`: The series, `observations × assets`.
  - $(arg_dict[:agg_obs])

# Returns

  - `B::Matrix{<:Real}`: The aggregated series, `div(observations, agg_obs) × assets`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_agg_series([1.0 2.0; 3.0 NaN; 5.0 6.0], 2)
1×2 Matrix{Float64}:
 2.0  2.0
```

# Related

  - [`ew_agg_vector`](@ref)
  - [`EWBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
"""
function ew_agg_series(A::AbstractMatrix{<:Real}, agg_obs::Integer)
    Tf = float(eltype(A))
    T, N = size(A)
    K = div(T, agg_obs)
    B = fill(Tf(NaN), K, N)
    for k in 1:K, i in 1:N
        s = zero(Tf)
        c = 0
        for t in ((k - 1) * agg_obs + 1):(k * agg_obs)
            a = A[t, i]
            if isfinite(a)
                s += a
                c += 1
            end
        end
        if !iszero(c)
            B[k, i] = s / c
        end
    end
    return B
end
"""
    ew_agg_vector(v::AbstractVector{<:Real}, agg_obs::Integer) -> Vector{<:Real}

Aggregate a series into one entry per complete window of consecutive observations.

This is [`ew_agg_series`](@ref) over one column, and it aggregates the market return and the reference return that ride beside the returns of the assets.

# Arguments

  - `v`: The series, one entry per observation.
  - $(arg_dict[:agg_obs])

# Returns

  - `w::Vector{<:Real}`: The aggregated series, of length `div(length(v), agg_obs)`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_agg_vector([1.0, 3.0, 5.0, 9.0], 2)
2-element Vector{Float64}:
 2.0
 7.0
```

# Related

  - [`ew_agg_series`](@ref)
  - [`EWBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
"""
function ew_agg_vector(v::AbstractVector{<:Real}, agg_obs::Integer)
    return vec(ew_agg_series(reshape(v, :, 1), agg_obs))
end
"""
    ew_beta_expand(Ba::AbstractMatrix{<:Real}, T::Integer, agg_obs::Integer) -> Matrix{<:Real}

Spread an aggregated beta series back over the observations it was aggregated from.

The recursion advances once per complete window, so every observation of a window reads the beta of the last window that closed at or before it. An observation before the first window closes reads `NaN`, because no beta has been estimated yet.

# Arguments

  - `Ba`: The aggregated betas, `windows × assets`.
  - `T`: Number of observations of the unaggregated series.
  - $(arg_dict[:agg_obs])

# Returns

  - `B::Matrix{<:Real}`: The betas, `T × assets`.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_beta_expand([1.0 2.0; 3.0 4.0], 5, 2)
5×2 Matrix{Float64}:
 NaN    NaN
   1.0    2.0
   1.0    2.0
   3.0    4.0
   3.0    4.0
```

# Related

  - [`ew_agg_series`](@ref)
  - [`EWBeta`](@ref)
  - [`EWMacroSensitivity`](@ref)
"""
function ew_beta_expand(Ba::AbstractMatrix{<:Real}, T::Integer,
                        agg_obs::Integer)::Matrix{<:Real}
    Tf = float(eltype(Ba))
    B = fill(Tf(NaN), T, size(Ba, 2))
    for t in 1:T
        k = div(t, agg_obs)
        if k >= 1
            B[t, :] = view(Ba, k, :)
        end
    end
    return B
end
"""
    ew_beta_residual_variance(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                              B::AbstractMatrix{<:Real}, decay::Real,
                              min_obs::Integer) -> Matrix{<:Real}

Run the exponentially weighted variance of the market-model residual of every asset.

The shrinkage of [`EWBeta`](@ref) weighs each raw beta against the noise of its own estimate, and this is that noise. Each residual is measured against the beta of the **previous** observation, so the residual carries no part of the beta it is about to correct. The recursion starts one observation after the warm-up ends, because a beta that has not been estimated yet leaves no residual.

# Arguments

  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - `B`: The raw betas, `observations × assets`, as [`ew_beta_series`](@ref) returns them.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])

# Returns

  - `Vr::Matrix{<:Real}`: The residual variance after each observation, `observations × assets`.

# Examples

```jldoctest
julia> B, Vm = PortfolioOptimisers.ew_beta_series([0.1 0.2; -0.1 0.3; 0.05 -0.05],
                                                  [0.1, -0.05, 0.02], 0.5, 1, 1e-12);

julia> PortfolioOptimisers.ew_beta_residual_variance([0.1 0.2; -0.1 0.3; 0.05 -0.05],
                                                     [0.1, -0.05, 0.02], B, 0.5, 1)
3×2 Matrix{Float64}:
 0.0          0.0
 0.00125      0.08
 0.000897222  0.0406722
```

# Related

  - [`EWBeta`](@ref)
  - [`ew_beta_series`](@ref)
  - [`ew_beta_shrink`](@ref)
"""
function ew_beta_residual_variance(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                                   B::AbstractMatrix{<:Real}, decay::Real, min_obs::Integer)
    Tf = float(eltype(B))
    K, N = size(X)
    Vr = zeros(Tf, K, N)
    v = zeros(Tf, N)
    om = one(Tf) - decay
    for k in 1:K
        if k - 1 >= min_obs
            for i in 1:N
                x = X[k, i]
                bp = B[k - 1, i]
                if isfinite(x) && isfinite(bp)
                    e = x - bp * rm[k]
                    v[i] = decay * v[i] + om * e * e
                end
            end
        end
        Vr[k, :] = v
    end
    return Vr
end
"""
    ew_masked_mean(v::AbstractVector{<:Real}, msk::AbstractVector{Bool}) -> Real

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
function ew_masked_mean(v::AbstractVector{<:Real}, msk::AbstractVector{Bool})::Real
    Tf = float(eltype(v))
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
    ew_masked_weighted_mean(v::AbstractVector{<:Real}, w::AbstractVector{<:Real},
                            msk::AbstractVector{Bool}) -> Real

Weighted mean of the entries a mask selects.

The shrinkage of [`EWBeta`](@ref) weights a group's mean beta by capitalisation, so a large asset carries the prior of its group further than a small one.

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
function ew_masked_weighted_mean(v::AbstractVector{<:Real}, w::AbstractVector{<:Real},
                                 msk::AbstractVector{Bool})::Real
    Tf = float(promote_type(eltype(v), eltype(w)))
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

The prior variance is the dispersion the betas actually show, less the dispersion their own estimation noise explains. What remains is the dispersion the group carries, and it is floored at zero because a group whose noise exceeds its spread carries none.

# Arguments

  - `b`: The raw betas.
  - `bev`: The estimation error variance of each beta.
  - `w`: The capitalisation weights.
  - `msk`: The mask that selects the group.

# Returns

  - `m::Real`: The capitalisation-weighted mean beta of the group.
  - `pv::Real`: The prior variance of the group.

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
                   min_group_size::Integer, bounds::Tuple{<:Real, <:Real},
                   min_val::Real) -> Vector{<:Real}

Shrink one cross-section of raw betas toward the capitalisation-weighted mean of its group.

A beta estimated from few observations is mostly noise, and the mean beta of the assets that share its industry is a better estimate of it than the number itself. The empirical Bayes weight is the share of the group's dispersion that is not noise, so a precise beta keeps its own value and a noisy one moves to the group.

# Mathematical definition

```math
\\begin{align}
\\beta_i^{\\ast} &= w_i \\beta_i + (1 - w_i) \\mu_g\\,, \\\\
w_i &= \\mathrm{clamp}\\!\\left(\\frac{\\tau_g^2}{\\tau_g^2 + \\sigma_i^2 + \\texttt{min\\_val}},\\, \\texttt{bounds}\\right)\\,, \\\\
\\sigma_i^2 &= \\frac{\\hat{\\sigma}_{\\varepsilon,i}^2}{n_{\\mathrm{eff}} \\left(V_m + \\texttt{min\\_val}\\right)}\\,.
\\end{align}
```

Where:

  - ``\\beta_i``: Raw beta of asset ``i``.
  - ``\\mu_g``: Capitalisation-weighted mean beta of the group of asset ``i``.
  - ``\\tau_g^2``: Prior variance of that group.
  - ``\\sigma_i^2``: Estimation error variance of ``\\beta_i``.
  - ``n_{\\mathrm{eff}}``: Effective sample size of the recursion, twice its half-life.
  - ``V_m``: Exponentially weighted variance of the market return.

# Algorithm

 1. Take the assets whose beta is not `NaN`, whose group label is set, and whose weight is finite and strictly positive. An asset outside that set keeps its raw beta.
 2. Estimate the mean and the prior variance of the whole cross-section, which every group below `min_group_size` falls back on.
 3. For each group, estimate its own mean and prior variance where it is large enough, and take the fallback where it is not.
 4. Weigh each raw beta against the noise of its own estimate, clamp the weight to `bounds`, and mix the raw beta with the mean of its group.

# Arguments

  - `b`: The raw betas.
  - `bev`: The estimation error variance of each beta.
  - `L`: The group label of each asset, [`CS_MISSING_GROUP`](@ref) where the asset carries none.
  - `w`: The capitalisation weights.
  - $(arg_dict[:min_group_size])
  - `bounds`: Lower and upper bound on the weight the raw beta keeps.
  - $(arg_dict[:min_val])

# Returns

  - `s::Vector{<:Real}`: The shrunk betas.

# Examples

```jldoctest
julia> PortfolioOptimisers.ew_beta_shrink([0.8, 1.4], [0.01, 0.01], [1, 1], [1.0, 1.0], 1,
                                          (0.0, 1.0), 1e-12)
2-element Vector{Float64}:
 0.8333333333362964
 1.3666666666637035
```

# Related

  - [`EWBeta`](@ref)
  - [`ew_beta_group_prior`](@ref)
  - [`ew_beta_residual_variance`](@ref)
  - [`cross_sectional_groups`](@ref)
  - [`CS_MISSING_GROUP`](@ref)
"""
function ew_beta_shrink(b::AbstractVector{<:Real}, bev::AbstractVector{<:Real},
                        L::AbstractVector{<:Integer}, w::AbstractVector{<:Real},
                        min_group_size::Integer, bounds::Tuple{<:Real, <:Real},
                        min_val::Real)
    s = [float(x) for x in b]
    vld = [!isnan(b[i]) &&
               L[i] != CS_MISSING_GROUP &&
               isfinite(w[i]) &&
               w[i] > zero(eltype(w)) for i in eachindex(b, L, w)]
    if !any(vld)
        return s
    end
    gm, gpv = ew_beta_group_prior(b, bev, w, vld)
    lo, hi = bounds
    for g in unique(L[i] for i in eachindex(L) if vld[i])
        msk = [vld[i] && L[i] == g for i in eachindex(L)]
        m, pv = if count(msk) < min_group_size
            (gm, gpv)
        else
            ew_beta_group_prior(b, bev, w, msk)
        end
        for i in eachindex(msk)
            if msk[i]
                q = clamp(pv / (pv + bev[i] + min_val), lo, hi)
                s[i] = q * b[i] + (one(q) - q) * m
            end
        end
    end
    return s
end
"""
    assert_ew_shrinkage_bounds(bounds::Tuple{<:Real, <:Real}) -> nothing

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
function assert_ew_shrinkage_bounds(bounds::Tuple{<:Real, <:Real})::Nothing
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
  - [`ew_agg_series`](@ref)
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

This is the archetype of every market beta Descriptor. The beta is the exponentially weighted covariance of an asset with the market, over the exponentially weighted variance of the market, and the market return is rebuilt from the Asset Panel by [`market_return_series`](@ref). Where the estimator names a categorical Panel Field in `group`, each raw beta is shrunk toward the capitalisation-weighted mean beta of its group by [`ew_beta_shrink`](@ref).

# Mathematical definition

```math
\\begin{align}
\\beta_{t,i} &= \\frac{C_{t,i}}{V_{t} + \\texttt{min\\_val}}\\,, \\\\
C_{t,i} &= \\lambda C_{t-1,i} + (1 - \\lambda)\\left(r_{t,i} - \\mu_{t-1,i}\\right)\\left(r_{m,t} - \\mu_{t-1}\\right)\\,, \\\\
V_{t} &= \\lambda V_{t-1} + (1 - \\lambda)\\left(r_{m,t} - \\mu_{t-1}\\right)^2\\,.
\\end{align}
```

Where:

  - ``r_{t,i}``: Return of asset ``i`` at observation ``t``.
  - ``r_{m,t}``: The market return at observation ``t``.
  - ``\\mu_{t,i}``, ``\\mu_{t}``: Exponentially weighted means of the asset and of the market.
  - ``\\lambda``: The decay factor.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWBeta(; mcap::AbstractString = "market_cap", decay::Real, min_obs::Integer,
           agg_obs::Integer = 1, group::Option{<:AbstractString} = nothing,
           min_group_size::Integer = 5,
           bounds::Tuple{<:Real, <:Real} = (0.0, 1.0),
           min_val::Real = 1e-12) -> EWBeta

Keywords correspond to the struct's fields. `decay` and `min_obs` take no default, because they depend on the data frequency: [`EWMarketBeta`](@ref) states a half-life instead and converts it through [`half_life_decay`](@ref) and [`half_life_min_obs`](@ref).

## Validation

  - `0 < decay < 1`.
  - `min_obs >= 1`.
  - `agg_obs >= 1`.
  - `min_group_size >= 1`.
  - `0 <= bounds[1] <= bounds[2] <= 1`.
  - `min_val > 0`.

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
          bounds ┼ Tuple{Float64, Float64}: (0.0, 1.0)
         min_val ┴ Float64: 1.0e-12
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWMarketBeta`](@ref)
  - [`EWDownsideBeta`](@ref)
  - [`ew_beta_series`](@ref)
  - [`ew_beta_shrink`](@ref)
  - [`market_return_series`](@ref)
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
    Name of the categorical Panel Field the shrinkage groups the cross-section by, or `nothing` to leave every beta raw.
    """
    group
    """
    $(field_dict[:min_group_size])
    """
    min_group_size
    """
    Lower and upper bound on the weight a shrunk beta keeps on its raw value. A bound of `(0, 1)` lets the data set the weight alone.
    """
    bounds
    """
    $(field_dict[:min_val])
    """
    min_val
    function EWBeta(mcap::AbstractString, decay::Real, min_obs::Integer, agg_obs::Integer,
                    group::Option{<:AbstractString}, min_group_size::Integer,
                    bounds::Tuple{<:Real, <:Real}, min_val::Real)
        assert_panel_terms(mcap, :mcap)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_ew_agg_obs(agg_obs)
        assert_nonempty_gt0_finite_val(min_group_size, :min_group_size)
        assert_ew_shrinkage_bounds(bounds)
        assert_nonempty_gt0_finite_val(min_val, :min_val)
        return new{typeof(mcap), typeof(decay), typeof(min_obs), typeof(agg_obs),
                   typeof(group), typeof(min_group_size), typeof(bounds), typeof(min_val)}(mcap,
                                                                                           decay,
                                                                                           min_obs,
                                                                                           agg_obs,
                                                                                           group,
                                                                                           min_group_size,
                                                                                           bounds,
                                                                                           min_val)
    end
end
function EWBeta(; mcap::AbstractString = "market_cap", decay::Real, min_obs::Integer,
                agg_obs::Integer = 1, group::Option{<:AbstractString} = nothing,
                min_group_size::Integer = 5, bounds::Tuple{<:Real, <:Real} = (0.0, 1.0),
                min_val::Real = 1e-12)::EWBeta
    return EWBeta(mcap, decay, min_obs, agg_obs, group, min_group_size, bounds, min_val)
end
"""
    ew_beta_output(group::Nothing, de::EWBeta, rd::ReturnsResult,
                   Ba::AbstractMatrix{<:Real}, Vm::AbstractVector{<:Real},
                   Xa::AbstractMatrix{<:Real}, rma::AbstractVector{<:Real}) -> Matrix{<:Real}
    ew_beta_output(group::AbstractString, de::EWBeta, rd::ReturnsResult,
                   Ba::AbstractMatrix{<:Real}, Vm::AbstractVector{<:Real},
                   Xa::AbstractMatrix{<:Real}, rma::AbstractVector{<:Real}) -> Matrix{<:Real}

Turn the recursion's raw betas into the Descriptor an [`EWBeta`](@ref) answers.

The `group` slot is read by dispatch: with no group the raw betas are spread over the observations they were aggregated from, and with a group each cross-section is shrunk first. A shrunk cross-section is recomputed only where a window closes, and it is held between windows, so the shrinkage runs on the same clock as the recursion.

# Arguments

  - `group`: The `group` slot of the estimator.
  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `Ba`: The raw betas, `windows × assets`.
  - `Vm`: The market variance after each window.
  - `Xa`: The aggregated returns, `windows × assets`.
  - `rma`: The aggregated market return, one entry per window.

# Returns

  - `B::Matrix{<:Real}`: The Descriptor before the active mask is applied, `observations × assets`.

# Related

  - [`EWBeta`](@ref)
  - [`descriptor`](@ref)
  - [`ew_beta_shrink`](@ref)
  - [`ew_beta_expand`](@ref)
  - [`cross_sectional_groups`](@ref)
"""
function ew_beta_output(::Nothing, de::EWBeta, rd::ReturnsResult,
                        Ba::AbstractMatrix{<:Real}, ::AbstractVector{<:Real},
                        ::AbstractMatrix{<:Real}, ::AbstractVector{<:Real})::Matrix{<:Real}
    return ew_beta_expand(Ba, size(rd.X, 1), de.agg_obs)
end
function ew_beta_output(group::AbstractString, de::EWBeta, rd::ReturnsResult,
                        Ba::AbstractMatrix{<:Real}, Vm::AbstractVector{<:Real},
                        Xa::AbstractMatrix{<:Real},
                        rma::AbstractVector{<:Real})::Matrix{<:Real}
    W = Matrix{float(eltype(rd.Z))}(panel_field_values(rd, de.mcap))
    L = cross_sectional_groups(descriptor_asset_panel(rd), rd.Z, group)
    Vr = ew_beta_residual_variance(Xa, rma, Ba, de.decay, de.min_obs)
    en = 2 * decay_half_life(de.decay)
    agg_obs = de.agg_obs
    min_obs = de.min_obs
    Tf = float(eltype(Ba))
    T, N = size(W)
    B = fill(Tf(NaN), T, N)
    s = fill(Tf(NaN), N)
    for t in 1:T
        k = div(t, agg_obs)
        if iszero(t % agg_obs) && k >= min_obs
            bev = view(Vr, k, :) ./ (en * (Vm[k] + de.min_val))
            s = ew_beta_shrink(view(Ba, k, :), bev, view(L, t, :), view(W, t, :),
                               de.min_group_size, de.bounds, de.min_val)
        end
        if k >= min_obs
            B[t, :] = s
        elseif k >= 1
            B[t, :] = view(Ba, k, :)
        end
    end
    return B
end
"""
    descriptor(de::EWBeta, rd::ReturnsResult) -> Matrix{<:Real}

Compute an exponentially weighted market beta Descriptor from a carrier.

# Algorithm

 1. Build the market return through [`market_return_series`](@ref), and mask the returns through [`ew_active_returns`](@ref).
 2. Aggregate the returns and the market return through [`ew_agg_series`](@ref) where `agg_obs` is greater than one.
 3. Run the recursion through [`ew_beta_series`](@ref).
 4. Shrink and spread the raw betas through [`ew_beta_output`](@ref), which reads the `group` slot by dispatch.
 5. Write `NaN` into the inactive cells through [`descriptor_active_fill!`](@ref).

An asset whose return is missing at an observation holds the beta it last carried, and the market state advances there whatever the asset does.

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
julia> res = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05], res...);

julia> descriptor(EWMarketBeta(; half_life = 1), rd)
3×2 Matrix{Float64}:
 0.6       1.2
 0.914432  0.982316
 1.00449   0.927219
```

# Related

  - [`EWBeta`](@ref)
  - [`EWMarketBeta`](@ref)
  - [`ew_beta_series`](@ref)
  - [`ew_beta_output`](@ref)
  - [`market_return_series`](@ref)
"""
function descriptor(de::EWBeta, rd::ReturnsResult)::Matrix{<:Real}
    pnl = descriptor_asset_panel(rd)
    rm = market_return_series(rd, de.mcap)
    X = ew_active_returns(rd.X, pnl)
    agg_obs = de.agg_obs
    Xa = isone(agg_obs) ? X : ew_agg_series(X, agg_obs)
    rma = isone(agg_obs) ? rm : ew_agg_vector(rm, agg_obs)
    Ba, Vm = ew_beta_series(Xa, rma, de.decay, de.min_obs, de.min_val)
    D = ew_beta_output(de.group, de, rd, Ba, Vm, Xa, rma)
    descriptor_active_fill!(D, pnl)
    return D
end
"""
    EWMarketBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                 decay::Real = half_life_decay(half_life),
                 min_obs::Integer = half_life_min_obs(half_life), agg_obs::Integer = 1,
                 group::Option{<:AbstractString} = nothing, min_group_size::Integer = 5,
                 bounds::Tuple{<:Real, <:Real} = (0.0, 1.0),
                 min_val::Real = 1e-12) -> EWBeta

Exponentially weighted sensitivity of an asset to the market portfolio.

The value is the beta of [`EWBeta`](@ref) against the capitalisation-weighted market return. It is the oldest measure of systematic risk there is: an asset of beta two moves twice as far as the market, and it earns the premium the market pays for carrying that move. The default half-life of `60` weights about as far back as a quarter of daily observations.

# Arguments

  - $(arg_dict[:mcap])
  - `half_life`: Half-life of the recursion, in observations. It fixes the defaults of `decay` and `min_obs`.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - $(arg_dict[:agg_obs])
  - `group`: Name of the categorical Panel Field the shrinkage groups by, or `nothing`.
  - $(arg_dict[:min_group_size])
  - `bounds`: Lower and upper bound on the weight a shrunk beta keeps on its raw value.
  - $(arg_dict[:min_val])

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
          bounds ┼ Tuple{Float64, Float64}: (0.0, 1.0)
         min_val ┴ Float64: 1.0e-12
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
                      bounds::Tuple{<:Real, <:Real} = (0.0, 1.0),
                      min_val::Real = 1e-12)::EWBeta
    return EWBeta(; mcap = mcap, decay = decay, min_obs = min_obs, agg_obs = agg_obs,
                  group = group, min_group_size = min_group_size, bounds = bounds,
                  min_val = min_val)
end
"""
    ew_macro_sensitivity_series(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                                rf::AbstractVector{<:Real}, decay::Real, min_obs::Integer,
                                min_val::Real) -> Matrix{<:Real}

Run the exponentially weighted partial beta of the returns on a reference series.

The partial beta is the coefficient of the reference series in the regression of the returns on the market **and** the reference series together, so it carries none of the exposure the market already explains. It is computed in closed form by the Frisch-Waugh decomposition, from the exponentially weighted moments alone and with no matrix to invert.

# Mathematical definition

```math
\\begin{align}
\\beta^{f}_{t,i} &= \\frac{C^{f}_{t,i} - C^{m}_{t,i} C^{mf}_{t} / \\tilde{V}^{m}_{t}}{V^{f}_{t} - \\left(C^{mf}_{t}\\right)^2 / \\tilde{V}^{m}_{t} + \\texttt{min\\_val}}\\,, \\\\
\\tilde{V}^{m}_{t} &= V^{m}_{t} + \\texttt{min\\_val}\\,.
\\end{align}
```

Where:

  - ``C^{m}_{t,i}``, ``C^{f}_{t,i}``: Exponentially weighted covariances of asset ``i`` with the market and with the reference series.
  - ``C^{mf}_{t}``: Exponentially weighted covariance of the market with the reference series.
  - ``V^{m}_{t}``, ``V^{f}_{t}``: Exponentially weighted variances of the market and of the reference series.

# Algorithm

 1. Skip the observation entirely where the reference return is not finite. The whole state freezes there, market included, because a partial beta needs both series at once.
 2. Take each deviation from the mean of the previous step, and update the mean after it.
 3. Write a fresh partial beta where the observation count of the panel and the valid count of the asset have both reached `min_obs`. An asset whose return is not valid keeps the value it last held.

# Arguments

  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - `rf`: The reference return, one entry per observation.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - $(arg_dict[:min_val])

# Returns

  - `B::Matrix{<:Real}`: The partial betas, `observations × assets`, before any mask is applied.

# Related

  - [`EWMacroSensitivity`](@ref)
  - [`descriptor`](@ref)
  - [`ew_beta_series`](@ref)
"""
function ew_macro_sensitivity_series(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                                     rf::AbstractVector{<:Real}, decay::Real,
                                     min_obs::Integer, min_val::Real)::Matrix{<:Real}
    Tf = float(promote_type(eltype(X), eltype(rm), eltype(rf)))
    T, N = size(X)
    B = fill(Tf(NaN), T, N)
    b = fill(Tf(NaN), N)
    mu = zeros(Tf, N)
    cam = zeros(Tf, N)
    caf = zeros(Tf, N)
    n = zeros(Int, N)
    mu_m = zero(Tf)
    mu_f = zero(Tf)
    var_m = zero(Tf)
    var_f = zero(Tf)
    cov_mf = zero(Tf)
    om = one(Tf) - decay
    c = 0
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
    return B
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted sensitivity of the returns to a reference series, after the market is removed.

An asset's exposure to an exchange rate, to a rate of interest, to inflation or to a basket of commodities is what remains after the market has taken its share of the move. The reference series is not a Panel Field, so it is not carried by the Asset Panel: [`descriptor`](@ref) takes it as the keyword `ref`, because carried input does not travel on a Result.

# Mathematical definition

```math
r_{t,i} = \\alpha_i + \\beta^{m}_i r_{m,t} + \\beta^{f}_i r_{f,t} + \\varepsilon_{t,i}\\,,
```

Where:

  - ``r_{m,t}``: The market return at observation ``t``.
  - ``r_{f,t}``: The reference return at observation ``t``.
  - ``\\beta^{f}_i``: The Descriptor, the partial sensitivity of asset ``i`` to the reference series.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWMacroSensitivity(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                       decay::Real = half_life_decay(half_life),
                       min_obs::Integer = half_life_min_obs(half_life),
                       agg_obs::Integer = 1, min_val::Real = 1e-12) -> EWMacroSensitivity

Keywords correspond to the struct's fields, except `half_life`, which is not a field: it fixes the defaults of `decay` and `min_obs`, and a value passed for either of those is used as it stands. The default half-life of `60` weights about as far back as a quarter of daily observations.

## Validation

  - `0 < decay < 1`.
  - `min_obs >= 1`.
  - `agg_obs >= 1`.
  - `min_val > 0`.

# Examples

```jldoctest
julia> EWMacroSensitivity(; half_life = 2)
EWMacroSensitivity
     mcap ┼ String: "market_cap"
    decay ┼ Float64: 0.7071067811865476
  min_obs ┼ Int64: 2
  agg_obs ┼ Int64: 1
  min_val ┴ Float64: 1.0e-12
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`EWBeta`](@ref)
  - [`ew_macro_sensitivity_series`](@ref)
  - [`market_return_series`](@ref)
"""
@concrete struct EWMacroSensitivity <: AbstractDescriptorEstimator
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
    $(field_dict[:min_val])
    """
    min_val
    function EWMacroSensitivity(mcap::AbstractString, decay::Real, min_obs::Integer,
                                agg_obs::Integer, min_val::Real)
        assert_panel_terms(mcap, :mcap)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_ew_agg_obs(agg_obs)
        assert_nonempty_gt0_finite_val(min_val, :min_val)
        return new{typeof(mcap), typeof(decay), typeof(min_obs), typeof(agg_obs),
                   typeof(min_val)}(mcap, decay, min_obs, agg_obs, min_val)
    end
end
function EWMacroSensitivity(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                            decay::Real = half_life_decay(half_life),
                            min_obs::Integer = half_life_min_obs(half_life),
                            agg_obs::Integer = 1, min_val::Real = 1e-12)::EWMacroSensitivity
    return EWMacroSensitivity(mcap, decay, min_obs, agg_obs, min_val)
end
"""
    descriptor(de::EWMacroSensitivity, rd::ReturnsResult;
               ref::Option{<:AbstractVector{<:Real}} = nothing) -> Matrix{<:Real}

Compute an exponentially weighted macro sensitivity Descriptor from a carrier and a reference series.

# Algorithm

 1. Build the market return through [`market_return_series`](@ref), and mask the returns through [`ew_active_returns`](@ref).
 2. Aggregate the returns, the market return and the reference return through [`ew_agg_series`](@ref) where `agg_obs` is greater than one.
 3. Run the recursion through [`ew_macro_sensitivity_series`](@ref).
 4. Spread the partial betas over the observations they were aggregated from, and write `NaN` into the inactive cells.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.
  - `ref`: The reference return, one entry per observation. It is required.

# Validation

  - `rd.pnl` is an [`AssetPanel`](@ref). Raises an [`IsNothingError`](@ref).
  - `ref` is not `nothing`. Raises an [`IsNothingError`](@ref).
  - `length(ref) == size(rd.X, 1)`. Raises a `DimensionMismatch`.
  - The rules of [`market_return_series`](@ref).

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> res = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05], res...);

julia> descriptor(EWMacroSensitivity(; half_life = 1), rd; ref = [0.02, -0.01, 0.03])
3×2 Matrix{Float64}:
  0.070978    0.141956
 15.2941    -10.5882
  1.96733    -1.21431
```

# Related

  - [`EWMacroSensitivity`](@ref)
  - [`ew_macro_sensitivity_series`](@ref)
  - [`market_return_series`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function descriptor(de::EWMacroSensitivity, rd::ReturnsResult;
                    ref::Option{<:AbstractVector{<:Real}} = nothing)::Matrix{<:Real}
    pnl = descriptor_asset_panel(rd)
    @argcheck(!isnothing(ref),
              IsNothingError("a macro sensitivity is measured against a reference return series that the Asset Panel does not carry, so `descriptor` takes it as the keyword `ref`, and it is nothing. Pass the series, one entry per observation."))
    X = ew_active_returns(rd.X, pnl)
    @argcheck(length(ref) == size(X, 1),
              DimensionMismatch("the reference return series carries one entry per observation, got length(ref) = $(length(ref)) and size(rd.X, 1) = $(size(X, 1))"))
    rm = market_return_series(rd, de.mcap)
    agg_obs = de.agg_obs
    Xa = isone(agg_obs) ? X : ew_agg_series(X, agg_obs)
    rma = isone(agg_obs) ? rm : ew_agg_vector(rm, agg_obs)
    rfa = isone(agg_obs) ? ref : ew_agg_vector(ref, agg_obs)
    Ba = ew_macro_sensitivity_series(Xa, rma, rfa, de.decay, de.min_obs, de.min_val)
    D = ew_beta_expand(Ba, size(X, 1), agg_obs)
    descriptor_active_fill!(D, pnl)
    return D
end
"""
    ew_downside_beta_series(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                            decay::Real, min_obs::Integer, mar::Real,
                            min_val::Real) -> Matrix{<:Real}

Run the exponentially weighted lower partial co-moment of the returns against the market.

The recursion advances at every observation, so an observation the market spends above the target adds nothing to the co-moment while the co-moments already there decay. That is what keeps the estimate moving through a calm market, which a recursion that advances only on a fall would freeze.

# Mathematical definition

```math
\\begin{align}
D_{t} &= \\min(r_{m,t} - \\mathrm{mar},\\, 0)\\,, \\quad D_{t,i} = \\min(r_{t,i} - \\mathrm{mar},\\, 0)\\,, \\\\
\\beta^{-}_{t,i} &= \\frac{\\lambda C_{t-1,i} + (1 - \\lambda) D_{t,i} D_{t}}{\\lambda V_{t-1} + (1 - \\lambda) D_{t}^2 + \\texttt{min\\_val}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{mar}``: The minimum acceptable return.
  - ``C_{t,i}``: Exponentially weighted co-moment of the shortfalls of asset ``i`` and of the market.
  - ``V_{t}``: Exponentially weighted second moment of the shortfall of the market.
  - ``\\lambda``: The decay factor.

# Arguments

  - `X`: The returns, `observations × assets`.
  - `rm`: The market return, one entry per observation.
  - $(arg_dict[:decay])
  - $(arg_dict[:min_obs])
  - `mar`: The minimum acceptable return.
  - $(arg_dict[:min_val])

# Returns

  - `B::Matrix{<:Real}`: The downside betas, `observations × assets`, before any mask is applied.

# Related

  - [`EWDownsideBeta`](@ref)
  - [`descriptor`](@ref)
  - [`ew_beta_series`](@ref)
"""
function ew_downside_beta_series(X::AbstractMatrix{<:Real}, rm::AbstractVector{<:Real},
                                 decay::Real, min_obs::Integer, mar::Real,
                                 min_val::Real)::Matrix{<:Real}
    Tf = float(promote_type(eltype(X), eltype(rm)))
    T, N = size(X)
    B = fill(Tf(NaN), T, N)
    cd = zeros(Tf, N)
    n = zeros(Int, N)
    vd = zero(Tf)
    om = one(Tf) - decay
    z = zero(Tf)
    for t in 1:T
        dm = min(rm[t] - mar, z)
        vd = decay * vd + om * dm * dm
        for i in 1:N
            x = X[t, i]
            if isfinite(x)
                cd[i] = decay * cd[i] + om * min(x - mar, z) * dm
                n[i] += 1
            end
            if t >= min_obs && n[i] >= min_obs
                B[t, i] = cd[i] / (vd + min_val)
            end
        end
    end
    return B
end
"""
$(DocStringExtensions.TYPEDEF)

Exponentially weighted sensitivity of the returns to the falls of the market, at every observation.

A beta that treats a rise and a fall alike says nothing about which of the two an asset follows. This Descriptor measures the second one alone, from the lower partial co-moment of the asset with the market: how far the asset falls when the market falls short of a target. An investor who fears a loss and not a gain reads this in place of the two-sided beta.

# Mathematical definition

```math
\\beta^{-}_{t,i} = \\frac{\\mathrm{EW}\\!\\left(D_{i} D_{m}\\right)}{\\mathrm{EW}\\!\\left(D_{m}^2\\right) + \\texttt{min\\_val}}\\,, \\quad D = \\min(r - \\mathrm{mar},\\, 0)\\,.
```

Where:

  - ``D_{i}``, ``D_{m}``: Shortfalls of asset ``i`` and of the market below the minimum acceptable return.
  - ``\\mathrm{mar}``: The minimum acceptable return.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EWDownsideBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                   decay::Real = half_life_decay(half_life),
                   min_obs::Integer = half_life_min_obs(half_life), mar::Real = 0.0,
                   min_val::Real = 1e-12) -> EWDownsideBeta

Keywords correspond to the struct's fields, except `half_life`, which is not a field: it fixes the defaults of `decay` and `min_obs`, and a value passed for either of those is used as it stands. The default half-life of `60` weights about as far back as a quarter of daily observations, and the default target of zero calls a loss the downside.

## Validation

  - `0 < decay < 1`.
  - `min_obs >= 1`.
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
  - [`ew_downside_beta_series`](@ref)
  - [`market_return_series`](@ref)
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
    Minimum acceptable return the shortfall of the asset and of the market are both measured below.
    """
    mar
    """
    $(field_dict[:min_val])
    """
    min_val
    function EWDownsideBeta(mcap::AbstractString, decay::Real, min_obs::Integer, mar::Real,
                            min_val::Real)
        assert_panel_terms(mcap, :mcap)
        assert_ew_decay(decay)
        assert_nonempty_gt0_finite_val(min_obs, :min_obs)
        assert_finite(mar, :mar)
        assert_nonempty_gt0_finite_val(min_val, :min_val)
        return new{typeof(mcap), typeof(decay), typeof(min_obs), typeof(mar),
                   typeof(min_val)}(mcap, decay, min_obs, mar, min_val)
    end
end
function EWDownsideBeta(; mcap::AbstractString = "market_cap", half_life::Real = 60.0,
                        decay::Real = half_life_decay(half_life),
                        min_obs::Integer = half_life_min_obs(half_life), mar::Real = 0.0,
                        min_val::Real = 1e-12)::EWDownsideBeta
    return EWDownsideBeta(mcap, decay, min_obs, mar, min_val)
end
"""
    descriptor(de::EWDownsideBeta, rd::ReturnsResult) -> Matrix{<:Real}

Compute an exponentially weighted downside beta Descriptor from a carrier.

# Algorithm

 1. Build the market return through [`market_return_series`](@ref), and mask the returns through [`ew_active_returns`](@ref).
 2. Run the recursion through [`ew_downside_beta_series`](@ref).
 3. Write `NaN` into the inactive cells through [`descriptor_active_fill!`](@ref).

An asset whose return is missing at an observation advances no co-moment of its own, and it still reads the market's shortfall of that observation once it is ready.

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
julia> res = asset_panel([NumericPanelInput(; name = \"market_cap\",
                                            vals = [1.0 2.0; 3.0 4.0; 5.0 6.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = [0.1 0.2; -0.1 0.0; 0.05 0.05], res...);

julia> descriptor(EWDownsideBeta(; half_life = 1), rd)
3×2 Matrix{Float64}:
 0.0      0.0
 2.33333  0.0
 2.33333  0.0
```

# Related

  - [`EWDownsideBeta`](@ref)
  - [`ew_downside_beta_series`](@ref)
  - [`market_return_series`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function descriptor(de::EWDownsideBeta, rd::ReturnsResult)::Matrix{<:Real}
    pnl = descriptor_asset_panel(rd)
    rm = market_return_series(rd, de.mcap)
    D = ew_downside_beta_series(ew_active_returns(rd.X, pnl), rm, de.decay, de.min_obs,
                                de.mar, de.min_val)
    descriptor_active_fill!(D, pnl)
    return D
end

export EWBeta, EWMarketBeta, EWMacroSensitivity, EWDownsideBeta
