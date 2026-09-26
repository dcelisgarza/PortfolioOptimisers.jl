"""
    const ATTRIBUTION_CURRENCY_FAMILY = "currency"

Family label whose rows carry no regression-estimation uncertainty.

A currency factor is a direct return rather than a coefficient the cross-sectional fit estimates, so the sandwich covariance says nothing about it. [`factor_attribution`](@ref) keeps such a row out of the Gram matrix and reports `NaN` for its standard error and for the standard error of its family.

# Related

  - [`factor_attribution`](@ref)
  - [`AttributionBreakdown`](@ref)
"""
const ATTRIBUTION_CURRENCY_FAMILY = "currency"
"""
    attribution_slice(B::MatNum, t::Integer)
    attribution_slice(B::Arr3Num, t::Integer)

Return the exposure slice of one observation.

A block whose exposures do not move holds one matrix, and that matrix is the slice of every observation. A block that keeps a history returns the slice of `t`. The shape of `B` selects the method, so a caller uses one function for both.

# Arguments

  - `B`: The static loadings, or the exposure history.
  - `t`: The observation.

# Returns

  - `Bt::MatNum`: The exposures of observation `t`, `assets × factors`.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_weights`](@ref)
"""
function attribution_slice(B::MatNum, ::Integer)
    return B
end
function attribution_slice(B::Arr3Num, t::Integer)
    return view(B, t, :, :)
end
"""
    attribution_weights(w::VecNum, t::Integer)
    attribution_weights(W::MatNum, t::Integer)

Return the portfolio weights of one observation.

A constant weight vector is the weights of every observation. A weight history returns the row of `t`. The shape of the weights selects the method, so a caller uses one function for both.

# Arguments

  - `w`: The constant weights, or the weight history.
  - `t`: The observation.

# Returns

  - `wt::VecNum`: The weights of observation `t`.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_slice`](@ref)
"""
function attribution_weights(w::VecNum, ::Integer)
    return w
end
function attribution_weights(W::MatNum, t::Integer)
    return view(W, t, :)
end
"""
    attribution_cov(x::VecNum, yc::VecNum)

Return the covariance of a series with a pre-centred series.

A realised attribution centres the portfolio series once and takes every covariance against it. The covariances of the parts with the whole then sum to the variance of the whole, so the components add up.

# Mathematical definition

```math
\\begin{align}
\\operatorname{cov}(x, y) &= \\frac{1}{T - 1} \\sum_{t=1}^{T} (x_{t} - \\bar{x})\\, y_{t}\\,, \\\\
0 &= \\sum_{t=1}^{T} y_{t}\\,.
\\end{align}
```

A centred ``y`` makes the sum equal to the sample covariance, because ``\\bar{x} \\sum_{t} y_{t}`` vanishes.

Where:

  - ``x_{t}``, ``y_{t}``: Entry ``t`` of the series and of the centred series.
  - ``\\bar{x}``: Mean of the series.
  - $(math_dict[:T])

# Arguments

  - `x`: The series.
  - `yc`: The pre-centred series.

# Returns

  - `cv::Real`: The covariance, with the corrected denominator.

# Related

  - [`factor_attribution`](@ref)
"""
function attribution_cov(x::VecNum, yc::VecNum)
    return LinearAlgebra.dot(yc, x .- mean(x)) / (length(yc) - 1)
end
"""
    attribution_align(rr::AbstractLoadingsRegressionResult, pr::AbstractPriorResult,
                      T::Integer)

Return the lag-aligned history a realised factor attribution reads off a factor model block.

The fit of observation `t` regresses the returns of `t` on the exposures of `t - lag`, so the exposures are trimmed at the tail and every return-like history at the head. A block whose exposures do not move keeps its one matrix, because a static slice needs no alignment.

A fit that warmed up on the first observations keeps fewer than the caller's series carries, so the two axes are lined up at their tail. The aligned history describes the last observations of the caller's series, and `rows` names them.

# Algorithm

 1. Read the exposure lag `lag`, the factor returns `f` and the idiosyncratic returns `eps` off the block, and the number of block observations `Tb`.
 2. Check that the caller's series is at least as long as the block, and that the block is longer than the lag.
 3. Take the block rows `brows = lag + 1:Tb`, whose returns an exposure explains, and the caller's rows `rows`, the last `Tb - lag` of its `T`. Aligned observation `j` therefore reads the exposures of block row `j`, the returns of block row `j + lag`, and the caller's row `T - Tb + lag + j`.
 4. Trim the exposure history by `lag` at its tail, giving `B`. A static loadings matrix is kept as it is.
 5. Take the rows `brows` of `f`, of `eps`, of the regression weights `rw` and of the idiosyncratic variances `vs`.
 6. Replace every non-finite entry of `B`, `f`, `eps`, `rw` and `vs` with zero.
 7. Slice the family re-basis to its first `Tb - lag` observations, giving `fcb`.

# Arguments

  - `rr`: The factor model block.
  - `pr`: The prior result the block travels on, which carries the two return series of a block that stores none.
  - `T`: The number of observations the caller's return series carries.

# Validation

  - The caller's series carries at least as many observations as the block, else a `DimensionMismatch` is raised.
  - The block carries more observations than the exposure lag, else a `DimensionMismatch` is raised.

# Returns

  - `B`: The exposures, static or lag-aligned.
  - `f::MatNum`: The lag-aligned factor returns.
  - `eps::MatNum`: The lag-aligned idiosyncratic returns.
  - `rw::Option{<:MatNum}`: The lag-aligned regression weights, or `nothing`.
  - `vs::Option{<:MatNum}`: The lag-aligned idiosyncratic variances, or `nothing`.
  - `fcb::Option{<:AbstractFactorFamilyBasis}`: The family re-basis over the aligned axis, or `nothing`.
  - `rows::UnitRange{Int}`: The rows of the caller's series the aligned history describes.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_lag`](@ref)
  - [`cs_regression_data`](@ref)
"""
function attribution_align(rr::AbstractLoadingsRegressionResult, pr::AbstractPriorResult,
                           T::Integer)
    lag = attribution_lag(rr)
    f = attribution_factor_returns(rr, pr)
    eps = attribution_idiosyncratic_returns(rr, pr)
    Tb = size(f, 1)
    @argcheck(T >= Tb,
              DimensionMismatch("the return series ($T observations) must carry at least as many observations as the factor model block ($Tb observations)"))
    @argcheck(Tb > lag,
              DimensionMismatch("the factor model block ($Tb observations) must carry more observations than the exposure lag ($lag)"))
    brows = (lag + 1):Tb
    rows = (T - Tb + lag + 1):T
    B = attribution_finite(attribution_trim_exposures(attribution_exposures(rr), lag))
    rw = attribution_trim_rows(attribution_regression_weights(rr), brows)
    vs = attribution_trim_rows(attribution_idiosyncratic_variances(rr), brows)
    fcb = attribution_trim_basis(attribution_family_basis(rr), Tb - lag)
    return (; B = B, f = attribution_finite(f[brows, :]),
            eps = attribution_finite(eps[brows, :]), rw = attribution_finite_rows(rw),
            vs = attribution_finite_rows(vs), fcb = fcb, rows = rows)
end
"""
    attribution_trim_exposures(B::MatNum, lag::Integer)
    attribution_trim_exposures(B::Arr3Num, lag::Integer)

Return the exposure history trimmed at the tail by the lag.

A static loadings matrix describes every observation, so it is returned unchanged. A history is trimmed at the tail, which is the half of the alignment the exposures carry.

# Arguments

  - `B`: The static loadings, or the exposure history.
  - `lag`: The exposure lag.

# Returns

  - `B`: The static loadings, or the trimmed exposure history.

# Related

  - [`attribution_align`](@ref)
"""
function attribution_trim_exposures(B::MatNum, ::Integer)
    return B
end
function attribution_trim_exposures(B::Arr3Num, lag::Integer)
    return B[1:(size(B, 1) - lag), :, :]
end
"""
    attribution_trim_rows(A::Nothing, rows)
    attribution_trim_rows(A::MatNum, rows)

Return the rows of an optional per-observation history, or `nothing`.

The regression weights and the idiosyncratic variances are optional, so the absent case is a method rather than a test at the call site.

# Arguments

  - `A`: The history, or `nothing`.
  - `rows`: The rows to keep.

# Returns

  - `A::Option{<:MatNum}`: The kept rows, or `nothing`.

# Related

  - [`attribution_align`](@ref)
"""
function attribution_trim_rows(::Nothing, ::Any)::Nothing
    return nothing
end
function attribution_trim_rows(A::MatNum, rows)
    return A[rows, :]
end
"""
    attribution_trim_basis(fcb::Nothing, n::Integer)
    attribution_trim_basis(fcb::FactorFamilyBasis, n::Integer)

Return the family re-basis over the aligned observation axis, or `nothing`.

The coefficients of observation `t` are coordinates in the basis of `t - lag`, so the basis follows the exposures and is sliced at the head of its own axis.

# Arguments

  - `fcb`: The family re-basis, or `nothing`.
  - `n`: The number of aligned observations.

# Returns

  - `fcb::Option{<:FactorFamilyBasis}`: The sliced basis, or `nothing`.

# Related

  - [`attribution_align`](@ref)
  - [`factor_basis_slice`](@ref)
"""
function attribution_trim_basis(::Nothing, ::Integer)::Nothing
    return nothing
end
function attribution_trim_basis(fcb::FactorFamilyBasis, n::Integer)
    return factor_basis_slice(fcb, 1:n)
end
"""
    attribution_finite_rows(A::Nothing)
    attribution_finite_rows(A::MatNum)

Return an optional per-observation history with every non-finite entry replaced by zero.

It pairs [`attribution_finite`](@ref) with the absent case, so the regression weights and the idiosyncratic variances take the same treatment as the histories beside them without a test at the call site.

# Arguments

  - `A`: The history, or `nothing`.

# Returns

  - `A::Option{<:MatNum}`: The history with its non-finite entries replaced by zero, or `nothing`.

# Related

  - [`attribution_align`](@ref)
  - [`attribution_finite`](@ref)
"""
function attribution_finite_rows(::Nothing)::Nothing
    return nothing
end
function attribution_finite_rows(A::MatNum)
    return attribution_finite(A)
end
"""
    attribution_window(al::NamedTuple, rows)

Return the aligned history restricted to one rolling window.

The alignment runs once over the whole history, and a window is a slice of the aligned axis. Windowing before the alignment would spend the lag again inside every window.

# Arguments

  - `al`: The aligned history.
  - `rows`: The rows of the aligned axis the window covers.

# Returns

  - `al::NamedTuple`: The aligned history over the window.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_align`](@ref)
"""
function attribution_window(al::NamedTuple, rows)
    return (; B = attribution_window_exposures(al.B, rows), f = al.f[rows, :],
            eps = al.eps[rows, :], rw = attribution_trim_rows(al.rw, rows),
            vs = attribution_trim_rows(al.vs, rows),
            fcb = attribution_window_basis(al.fcb, rows), rows = rows)
end
"""
    attribution_window_exposures(B::MatNum, rows)
    attribution_window_exposures(B::Arr3Num, rows)

Return the exposures of one rolling window.

A static loadings matrix describes every window, so it is returned unchanged.

# Arguments

  - `B`: The static loadings, or the aligned exposure history.
  - `rows`: The rows the window covers.

# Returns

  - `B`: The static loadings, or the exposures of the window.

# Related

  - [`attribution_window`](@ref)
"""
function attribution_window_exposures(B::MatNum, ::Any)
    return B
end
function attribution_window_exposures(B::Arr3Num, rows)
    return B[rows, :, :]
end
"""
    attribution_window_basis(fcb::Nothing, rows)
    attribution_window_basis(fcb::FactorFamilyBasis, rows)

Return the family re-basis of one rolling window, or `nothing`.

# Arguments

  - `fcb`: The aligned family re-basis, or `nothing`.
  - `rows`: The rows the window covers.

# Returns

  - `fcb::Option{<:FactorFamilyBasis}`: The sliced basis, or `nothing`.

# Related

  - [`attribution_window`](@ref)
  - [`factor_basis_slice`](@ref)
"""
function attribution_window_basis(::Nothing, ::Any)::Nothing
    return nothing
end
function attribution_window_basis(fcb::FactorFamilyBasis, rows)
    return factor_basis_slice(fcb, rows)
end
"""
    attribution_window_weights(w::VecNum, rows)
    attribution_window_weights(W::MatNum, rows)

Return the portfolio weights of one rolling window.

A constant weight describes every window, so it is returned unchanged.

# Arguments

  - `w`: The constant weights, or the weight history.
  - `rows`: The rows the window covers.

# Returns

  - `w`: The constant weights, or the weights of the window.

# Related

  - [`attribution_window`](@ref)
"""
function attribution_window_weights(w::VecNum, ::Any)
    return w
end
function attribution_window_weights(W::MatNum, rows)
    return W[rows, :]
end
"""
    attribution_weight_moments(w::VecNum, ::Integer)
    attribution_weight_moments(W::MatNum, n::Integer)

Return the mean weight of each asset and the spread of its history.

A constant weight has no spread, so the spread is `nothing` and a reader dispatches on it rather than reading a vector of zeros.

# Mathematical definition

```math
\\begin{align}
\\bar{w}_{i} &= \\frac{1}{T} \\sum_{t=1}^{T} w_{ti}\\,, \\\\
s_{i} &= \\operatorname{sd}(w_{\\cdot i})\\,.
\\end{align}
```

Where:

  - ``\\bar{w}_{i}``, ``s_{i}``: Mean weight and weight spread of asset ``i``.
  - $(math_dict[:w_t_att])
  - $(math_dict[:cov_sd_att])
  - $(math_dict[:T])

# Arguments

  - `w`: The constant weights, or the weight history.
  - `n`: The number of assets.

# Returns

  - `weight::VecNum`: The mean weight of each asset.
  - `weight_std::Option{<:VecNum}`: The spread of each asset's weight, or `nothing`.

# Related

  - [`factor_attribution`](@ref)
  - [`AssetAttributionBreakdown`](@ref)
"""
function attribution_weight_moments(w::VecNum, ::Integer)
    return w, nothing
end
function attribution_weight_moments(W::MatNum, ::Integer)
    return vec(mean(W; dims = 1)), vec(std(W; dims = 1))
end
"""
    realised_attribution(W::VecNum_MatNum, ret::VecNum, al::NamedTuple,
                         fam::Option{<:VecStr}, assets::Bool, se::Bool, ppy::Number)
        -> FactorAttributionResult

Decompose one realised return series over the factors of an aligned factor model history.

Every realised method of [`factor_attribution`](@ref), and every window of every rolling method, calls this function. It assumes the alignment is done, so it reads the exposures, the factor returns and the idiosyncratic returns as they stand.

# Mathematical definition

The portfolio return splits into three series:

```math
\\begin{align}
s_{t} &= \\boldsymbol{g}_{t}^{\\intercal} \\boldsymbol{f}_{t} = \\sum_{k=1}^{K} g_{tk} f_{tk}\\,, \\\\
e_{t} &= \\boldsymbol{w}_{t}^{\\intercal} \\boldsymbol{\\varepsilon}_{t}\\,, \\\\
u_{t} &= r_{t} - s_{t} - e_{t}\\,.
\\end{align}
```

The systematic, the idiosyncratic and the unattributed components are ``s``, ``e`` and ``u``, and each component is the five numbers of its series. The total is ``r`` itself, so its volatility contribution is ``\\sqrt{p}\\, \\sigma_{P}``, its variance share is ``1`` and its correlation is ``1``. The three series sum to ``r``, and the covariance is linear, so the three volatility contributions sum to the total and the three variance shares sum to ``1``.

Row ``k`` of the factor axis takes its numbers from the series ``g_{\\cdot k} f_{\\cdot k}``. These series sum to ``s`` over ``k``, so the rows sum to the systematic component.

```math
\\begin{align}
\\bar{g}_{k} &= \\frac{1}{T} \\sum_{t=1}^{T} g_{tk}\\,, \\\\
\\mathrm{VC}_{k} &= \\mathrm{VC}(g_{\\cdot k} f_{\\cdot k})\\,, \\\\
\\mathrm{PV}_{k} &= \\mathrm{PV}(g_{\\cdot k} f_{\\cdot k})\\,, \\\\
\\mathrm{MC}_{k} &= \\mathrm{MC}(g_{\\cdot k} f_{\\cdot k})\\,, \\\\
\\rho_{k} &= \\frac{\\operatorname{cov}(f_{\\cdot k}, r)}{\\operatorname{sd}(f_{\\cdot k})\\, \\sigma_{P}}\\,.
\\end{align}
```

The row also carries the exposure spread ``\\operatorname{sd}(g_{\\cdot k})``, the factor volatility ``\\sqrt{p}\\, \\operatorname{sd}(f_{\\cdot k})`` and the factor mean return ``p\\, \\bar{f}_{k}``. The correlation ``\\rho_{k}`` is of the factor return and not of its contribution, and it is `NaN` when ``\\operatorname{sd}(f_{\\cdot k})`` is zero.

Where:

  - ``u_{t}``: Unattributed return of the portfolio at observation ``t``.
  - $(math_dict[:s_e_t_att])
  - ``\\bar{g}_{k}``, ``\\bar{f}_{k}``: Mean portfolio exposure to factor ``k``, and mean return of factor ``k``.
  - ``\\mathrm{VC}_{k}``, ``\\mathrm{PV}_{k}``, ``\\mathrm{MC}_{k}``, ``\\rho_{k}``: Volatility contribution, variance share, mean return contribution and correlation with the portfolio of factor ``k``.
  - $(math_dict[:g_t_att])
  - $(math_dict[:f_t_att])
  - $(math_dict[:w_t_att])
  - $(math_dict[:eps_t_att])
  - $(math_dict[:r_t_att])
  - $(math_dict[:sigma_P_att])
  - $(math_dict[:VC_att])
  - $(math_dict[:cov_sd_att])
  - $(math_dict[:p_ppy])
  - $(math_dict[:K])
  - $(math_dict[:T])

# Algorithm

 1. Centre the portfolio series once, and take every covariance against it.
 2. Form the per-observation portfolio exposure `g` and the per-observation systematic and idiosyncratic series.
 3. Take each component's mean, volatility and covariance with the portfolio.
 4. Put the difference between the portfolio series and the model's reconstruction into the remainder.
 5. Sum the factor rows by family, and over the assets when `assets = true`.
 6. Scale by `ppy`.

# Arguments

  - `W`: The constant weights, or the weight history.
  - `ret`: The net portfolio return series.
  - `al`: The aligned factor model history.
  - `fam`: The family label of each raw factor, or `nothing`.
  - `assets`: Whether to fill the asset axis and the asset-by-factor matrices.
  - `se`: Whether to fill the standard errors of the mean return contributions.
  - `ppy`: Periods per year the numbers are scaled to.

# Validation

  - The portfolio volatility is positive, else a `DomainError` is raised.
  - `ppy > 0`, else a `DomainError` is raised.

# Returns

  - `fa::FactorAttributionResult`: The attribution of the series.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_align`](@ref)
  - [`attribution_standard_errors`](@ref)
"""
function realised_attribution(W::VecNum_MatNum, ret::VecNum, al::NamedTuple,
                              fam::Option{<:VecStr}, assets::Bool, se::Bool,
                              ppy::Number)::FactorAttributionResult
    sc = attribution_scale(ppy)
    f, eps = al.f, al.eps
    T, K = size(f)
    N = size(eps, 2)
    total_mu = mean(ret)
    total_vol = std(ret)
    @argcheck(total_vol > zero(total_vol),
              DomainError(total_vol,
                          "the portfolio return series must have a positive volatility for an attribution to divide by it"))
    retc = ret .- total_mu
    Tf = promote_type(real(eltype(f)), real(eltype(ret)), real(eltype(W)),
                      real(eltype(al.B)))
    g = Matrix{Tf}(undef, T, K)
    sysr = Matrix{Tf}(undef, T, N)
    for t in 1:T
        Bt = attribution_slice(al.B, t)
        g[t, :] = transpose(Bt) * attribution_weights(W, t)
        sysr[t, :] = Bt * view(f, t, :)
    end
    fpnl = g .* f
    sys_pnl = vec(sum(fpnl; dims = 2))
    idio_pnl = [LinearAlgebra.dot(attribution_weights(W, t), view(eps, t, :)) for t in 1:T]
    unattr_pnl = ret .- sys_pnl .- idio_pnl
    f_vol = vec(std(f; dims = 1))
    f_cov = [attribution_cov(view(f, :, k), retc) for k in 1:K]
    f_var_contrib = [attribution_cov(view(fpnl, :, k), retc) for k in 1:K]
    sers = se ? attribution_standard_errors(g, al, fam, sc.s1, T) : attribution_no_errors()
    sys = attribution_series_component(sys_pnl, retc, total_vol, sc, sers.sys)
    idio = attribution_series_component(idio_pnl, retc, total_vol, sc, sers.sys)
    unattr = attribution_series_component(unattr_pnl, retc, total_vol, sc, nothing)
    total = AttributionComponent(total_vol * sc.s2, total_vol * sc.s2, one(total_vol),
                                 total_mu * sc.s1, one(total_vol), nothing)
    fbd = AttributionBreakdown(nothing, vec(mean(g; dims = 1)), vec(std(g; dims = 1)),
                               f_vol .* sc.s2,
                               [attribution_safe_corr(f_cov[k], f_vol[k], total_vol)
                                for k in 1:K], f_var_contrib ./ total_vol .* sc.s2,
                               f_var_contrib ./ total_vol^2,
                               vec(mean(f; dims = 1)) .* sc.s1,
                               vec(mean(fpnl; dims = 1)) .* sc.s1, sers.factor)
    fmbd = attribution_family_axis(fam, fbd, attribution_family_spread(fam, g), sers.family)
    abd, afc = realised_attribution_assets(assets, W, al, sysr, retc, total_vol, sc)
    return FactorAttributionResult(sys, idio, unattr, total, fbd, fmbd, abd, afc, true, ppy)
end
"""
    attribution_series_component(pnl::VecNum, retc::VecNum, total_vol::Number,
                                 sc::NamedTuple, mu_se) -> AttributionComponent

Return one component of a realised attribution from its own return series.

The systematic, the idiosyncratic and the unattributed series are decomposed alike, so one verb builds all three. The volatility contribution is the covariance with the portfolio over the portfolio volatility, which is what makes the three sum to the portfolio volatility exactly.

# Mathematical definition

```math
\\begin{align}
\\mathrm{vol}(c) &= \\sqrt{p}\\, \\operatorname{sd}(c)\\,, \\\\
\\mathrm{VC}(c) &= \\sqrt{p}\\, \\frac{\\operatorname{cov}(c, r)}{\\sigma_{P}}\\,, \\\\
\\mathrm{PV}(c) &= \\frac{\\operatorname{cov}(c, r)}{\\sigma_{P}^{2}}\\,, \\\\
\\mathrm{MC}(c) &= \\frac{p}{T} \\sum_{t=1}^{T} c_{t}\\,, \\\\
\\rho(c) &= \\frac{\\operatorname{cov}(c, r)}{\\operatorname{sd}(c)\\, \\sigma_{P}}\\,.
\\end{align}
```

``\\rho(c)`` is `NaN` when ``\\operatorname{sd}(c)`` is zero. For series that sum to ``r``, the covariances sum to ``\\sigma_{P}^{2}``, so the volatility contributions sum to ``\\sqrt{p}\\, \\sigma_{P}`` and the variance shares sum to ``1``.

Where:

  - ``c_{t}``: Return of the series at observation ``t``.
  - ``\\mathrm{vol}(c)``: Standalone volatility of the series.
  - $(math_dict[:VC_att])
  - $(math_dict[:r_t_att])
  - $(math_dict[:sigma_P_att])
  - $(math_dict[:cov_sd_att])
  - $(math_dict[:p_ppy])
  - $(math_dict[:T])

# Arguments

  - `pnl`: The component's own return series.
  - `retc`: The centred portfolio return series.
  - `total_vol`: The portfolio volatility.
  - `sc`: The two annualisation factors.
  - `mu_se`: The standard error of the mean return contribution, or `nothing`.

# Returns

  - `c::AttributionComponent`: The component.

# Related

  - [`realised_attribution`](@ref)
  - [`AttributionComponent`](@ref)
"""
function attribution_series_component(pnl::VecNum, retc::VecNum, total_vol::Number,
                                      sc::NamedTuple, mu_se)
    vol = std(pnl)
    cv = attribution_cov(pnl, retc)
    return AttributionComponent(vol * sc.s2, cv / total_vol * sc.s2, cv / total_vol^2,
                                mean(pnl) * sc.s1,
                                attribution_safe_corr(cv, vol, total_vol), mu_se)
end
"""
    attribution_family_spread(fam::Nothing, g::MatNum)
    attribution_family_spread(fam::VecStr, g::MatNum)

Return the spread of each family's portfolio exposure over the observations.

A family's exposure is the sum of the exposures of its factors, so its spread is the spread of that sum and not the sum of the spreads.

# Mathematical definition

```math
\\begin{align}
s_{\\mathcal{F}} &= \\operatorname{sd}\\left(\\sum_{k \\in \\mathcal{F}} g_{\\cdot k}\\right)\\,.
\\end{align}
```

Where:

  - ``s_{\\mathcal{F}}``: Exposure spread of the family.
  - $(math_dict[:F_fam_att])
  - $(math_dict[:g_t_att])
  - $(math_dict[:cov_sd_att])

# Arguments

  - `fam`: The family label of each raw factor, or `nothing`.
  - `g`: The per-observation portfolio exposure, `observations × factors`.

# Returns

  - `exposure_std::Option{<:VecNum}`: The spread of each family's exposure, or `nothing`.

# Related

  - [`realised_attribution`](@ref)
  - [`attribution_family_axis`](@ref)
"""
function attribution_family_spread(::Nothing, ::MatNum)::Nothing
    return nothing
end
function attribution_family_spread(fam::VecStr, g::MatNum)
    fi = attribution_family_index(fam)
    return [std(vec(sum(view(g, :, i); dims = 2))) for i in fi.idx]
end
"""
    attribution_no_errors()

Return the empty standard errors of an attribution that was not asked for them.

The three answers are `nothing` together, so a Result built without `se = true` carries `nothing` in every place a standard error would sit.

# Returns

  - `sys::Nothing`: The systematic standard error.
  - `factor::Nothing`: The per-factor standard errors.
  - `family::Nothing`: The per-family standard errors.

# Related

  - [`realised_attribution`](@ref)
  - [`attribution_standard_errors`](@ref)
"""
function attribution_no_errors()
    return (; sys = nothing, factor = nothing, family = nothing)
end
"""
    attribution_standard_errors(g::MatNum, al::NamedTuple, fam::Option{<:VecStr},
                                s1::Number, T::Integer) -> NamedTuple

Return the standard errors of the mean return contributions of a realised attribution.

The cross-sectional fit estimates the factor returns of every observation, and that estimation error propagates into every mean return contribution. The error is the sandwich covariance of the fit, summed over the observations through the portfolio's own exposure.

Under a family re-basis the raw Gram matrix is singular by construction, so the sandwich is taken in the reduced full-rank basis and mapped back onto the raw axis for the per-factor and per-family answers. The systematic answer is invariant under the change of basis, so it is read in the reduced basis directly.

The systematic and the idiosyncratic errors are equal. The portfolio return is observed, so the two estimation errors sum to zero.

# Mathematical definition

```math
\\begin{align}
\\mathrm{SE} &= \\frac{p}{T} \\sqrt{\\sum_{t=1}^{T} \\boldsymbol{g}_{t}^{\\intercal} \\mathbf{V}_{t} \\boldsymbol{g}_{t}}\\,, \\\\
\\mathrm{SE}_{k} &= \\frac{p}{T} \\sqrt{\\sum_{t=1}^{T} g_{tk}^{2}\\, V_{t,kk}}\\,.
\\end{align}
```

[`attribution_sandwich`](@ref) states ``\\mathbf{V}_{t}``. The systematic error ``\\mathrm{SE}`` is the error of the mean of ``s_{t} = \\boldsymbol{g}_{t}^{\\intercal} \\hat{\\boldsymbol{f}}_{t}``, with the estimation errors of two observations taken as independent. A factor of the currency family is not estimated, so its row and its column of ``\\mathbf{V}_{t}`` are zero, and ``\\mathrm{SE}_{k}`` of such a factor is `NaN`.

Where:

  - ``\\mathrm{SE}``, ``\\mathrm{SE}_{k}``: Standard error of the systematic mean return contribution, and of the mean return contribution of factor ``k``.
  - ``\\hat{\\boldsymbol{f}}_{t}``: Factor returns the cross-sectional fit estimates at observation ``t``.
  - ``V_{t,kk}``: Diagonal entry ``k`` of ``\\mathbf{V}_{t}``.
  - $(math_dict[:g_t_att])
  - $(math_dict[:V_t_att])
  - $(math_dict[:p_ppy])
  - $(math_dict[:T])

# Arguments

  - `g`: The per-observation portfolio exposure on the raw axis, `observations × factors`.
  - `al`: The aligned factor model history.
  - `fam`: The family label of each raw factor, or `nothing`.
  - `s1`: The factor a mean takes under the annualisation.
  - `T`: The number of aligned observations.

# Validation

  - `al.rw` and `al.vs` are not `nothing`, else an `IsNothingError` names the field.

# Returns

  - `sys::Real`: The standard error of the systematic mean return contribution.
  - `factor::VecNum`: The standard error of each factor's mean return contribution.
  - `family::Option{<:VecNum}`: The standard error of each family's, or `nothing`.

# Related

  - [`factor_attribution`](@ref)
  - [`realised_attribution`](@ref)
  - [`ATTRIBUTION_CURRENCY_FAMILY`](@ref)
"""
function attribution_standard_errors(g::MatNum, al::NamedTuple, fam::Option{<:VecStr},
                                     s1::Number, T::Integer)
    rw = assert_attribution_field(al.rw, :rw)
    vs = assert_attribution_field(al.vs, :vs)
    K = size(g, 2)
    cur = attribution_currency_mask(fam, K)
    red = attribution_reduce_for_errors(al.fcb, al.B, g, fam, T)
    keep = findall(!, red.currency)
    se(v) = s1 * sqrt(max(zero(v), v)) / T
    V = [attribution_sandwich(view(red.B, t, :, :), view(rw, t, :), view(vs, t, :), keep)
         for t in 1:T]
    sys = se(sum(LinearAlgebra.dot(view(red.g, t, keep), V[t], view(red.g, t, keep))
                 for t in 1:T))
    Vf = [attribution_expand_errors(al.fcb, V[t], keep, red.nr, t) for t in 1:T]
    factor = [se(sum(g[t, k]^2 * Vf[t][k, k] for t in 1:T)) for k in 1:K]
    for k in cur
        factor[k] = oftype(factor[k], NaN)
    end
    return (; sys = sys, factor = factor,
            family = attribution_family_errors(fam, g, Vf, s1, T))
end
"""
    attribution_currency_mask(fam::Nothing, K::Integer)
    attribution_currency_mask(fam::VecStr, K::Integer)

Return the raw factors whose family is the currency family.

# Arguments

  - `fam`: The family label of each raw factor, or `nothing`.
  - `K`: The number of raw factors.

# Returns

  - `cur::Vector{Int}`: The raw factors of the currency family.

# Related

  - [`attribution_standard_errors`](@ref)
  - [`ATTRIBUTION_CURRENCY_FAMILY`](@ref)
"""
function attribution_currency_mask(::Nothing, ::Integer)
    return Int[]
end
function attribution_currency_mask(fam::VecStr, ::Integer)
    return findall(f -> String(f) == ATTRIBUTION_CURRENCY_FAMILY, fam)
end
"""
    attribution_reduce_for_errors(fcb::Nothing, B, g::MatNum, fam, T::Integer)
    attribution_reduce_for_errors(fcb::FactorFamilyBasis, B, g::MatNum, fam, T::Integer)

Return the exposures and the portfolio exposure the sandwich covariance is taken in.

A block that constrains no family is regressed on its raw axis, and the function returns the two inputs unchanged. A block that constrains a family has a singular raw Gram matrix, so the sandwich is taken in the reduced full-rank basis.

# Arguments

  - `fcb`: The family re-basis, or `nothing`.
  - `B`: The static loadings, or the aligned exposure history.
  - `g`: The per-observation portfolio exposure on the raw axis.
  - `fam`: The family label of each raw factor, or `nothing`.
  - `T`: The number of aligned observations.

# Returns

  - `B::Arr3Num`: The exposures of the regression basis, one slice per observation.
  - `g::MatNum`: The portfolio exposure in the regression basis.
  - `currency::Vector{Bool}`: Whether each column of the regression basis is a currency factor.
  - `nr::Int`: The number of columns of the regression basis.

# Related

  - [`attribution_standard_errors`](@ref)
  - [`reduce_exposures`](@ref)
  - [`project_factor_coordinates`](@ref)
"""
function attribution_reduce_for_errors(::Nothing, B, g::MatNum, fam::Option{<:VecStr},
                                       T::Integer)
    K = size(g, 2)
    return (; B = attribution_broadcast_exposures(B, T), g = g,
            currency = attribution_currency_flags(fam, K), nr = K)
end
function attribution_reduce_for_errors(fcb::FactorFamilyBasis, B, g::MatNum,
                                       fam::Option{<:VecStr}, T::Integer)
    Br = reduce_exposures(fcb, attribution_broadcast_exposures(B, T))
    gr = project_factor_coordinates(fcb, g)
    famr = attribution_reduce_families(fcb, fam)
    return (; B = Br, g = gr, currency = attribution_currency_flags(famr, size(Br, 3)),
            nr = size(Br, 3))
end
"""
    attribution_reduce_families(fcb::FactorFamilyBasis, fam::Nothing)
    attribution_reduce_families(fcb::FactorFamilyBasis, fam::VecStr)

Return the family label of each column of the reduced factor axis.

# Arguments

  - `fcb`: The family re-basis.
  - `fam`: The family label of each raw factor, or `nothing`.

# Returns

  - `fam::Option{<:VecStr}`: The family label of each reduced factor, or `nothing`.

# Related

  - [`attribution_reduce_for_errors`](@ref)
  - [`reduce_factor_names`](@ref)
"""
function attribution_reduce_families(::FactorFamilyBasis, ::Nothing)::Nothing
    return nothing
end
function attribution_reduce_families(fcb::FactorFamilyBasis, fam::VecStr)
    return String[String(fam[k]) for k in retained_factor_indices(fcb)]
end
"""
    attribution_currency_flags(fam::Nothing, K::Integer)
    attribution_currency_flags(fam::VecStr, K::Integer)

Return whether each factor of an axis belongs to the currency family.

# Arguments

  - `fam`: The family label of each factor of the axis, or `nothing`.
  - `K`: The number of factors of the axis.

# Returns

  - `flags::Vector{Bool}`: Whether each factor belongs to the currency family.

# Related

  - [`attribution_reduce_for_errors`](@ref)
  - [`ATTRIBUTION_CURRENCY_FAMILY`](@ref)
"""
function attribution_currency_flags(::Nothing, K::Integer)
    return falses(K)
end
function attribution_currency_flags(fam::VecStr, ::Integer)
    return Bool[String(f) == ATTRIBUTION_CURRENCY_FAMILY for f in fam]
end
"""
    attribution_broadcast_exposures(B::MatNum, T::Integer)
    attribution_broadcast_exposures(B::Arr3Num, T::Integer)

Return an exposure history of `T` slices, repeating a static loadings matrix.

The sandwich covariance runs per observation, and the basis transforms it reads take a history, so a static block is repeated once here rather than branched at every use.

# Arguments

  - `B`: The static loadings, or the aligned exposure history.
  - `T`: The number of aligned observations.

# Returns

  - `Ms::Arr3Num`: The exposure history, `observations × assets × factors`.

# Related

  - [`attribution_reduce_for_errors`](@ref)
"""
function attribution_broadcast_exposures(B::MatNum, T::Integer)
    N, K = size(B)
    Ms = Array{eltype(B), 3}(undef, T, N, K)
    for k in 1:K, i in 1:N, t in 1:T
        Ms[t, i, k] = B[i, k]
    end
    return Ms
end
function attribution_broadcast_exposures(B::Arr3Num, ::Integer)
    return B
end
"""
    attribution_sandwich(Bt::MatNum, q::VecNum, s2::VecNum, keep::AbstractVector{<:Integer})

Return the sandwich covariance of the factor returns of one observation.

The fit of one observation is a weighted least squares across the assets, so the covariance of its coefficients is the sandwich of the Gram matrix around the weighted idiosyncratic variances. A rank-deficient Gram matrix falls back to the pseudo-inverse, which is the policy [`PseudoInverseFallback`](@ref) states and the rank test [`cross_sectional_rank`](@ref) applies, so the standard errors of a collinear cross-section are the minimum-norm answer rather than an arbitrarily large one.

# Mathematical definition

```math
\\begin{align}
\\mathbf{G}_{t} &= \\mathbf{B}_{t}^{\\intercal} \\mathbf{Q}_{t} \\mathbf{B}_{t}\\,, \\\\
\\mathbf{V}_{t} &= \\mathbf{G}_{t}^{+} \\mathbf{B}_{t}^{\\intercal} \\mathbf{Q}_{t} \\boldsymbol{\\Omega}_{t} \\mathbf{Q}_{t} \\mathbf{B}_{t} \\mathbf{G}_{t}^{+}\\,.
\\end{align}
```

``\\mathbf{B}_{t}`` here holds the columns of the estimated factors alone. ``\\mathbf{G}_{t}^{+}`` is ``\\mathbf{G}_{t}^{-1}`` when ``\\mathbf{G}_{t}`` has full rank, and the Moore-Penrose pseudo-inverse when it does not.

Where:

  - ``\\mathbf{G}_{t}``: Gram matrix of the weighted cross-sectional fit of observation ``t``.
  - ``\\mathbf{G}_{t}^{+}``: Pseudo-inverse of ``\\mathbf{G}_{t}``.
  - $(math_dict[:B_t_att])
  - $(math_dict[:Q_t_att])
  - $(math_dict[:Omega_t_att])
  - $(math_dict[:V_t_att])

# Arguments

  - `Bt`: The exposures of the observation, `assets × factors`.
  - `q`: The regression weight of each asset at the observation.
  - `s2`: The idiosyncratic variance of each asset at the observation.
  - `keep`: The factors the regression estimates.

# Returns

  - `V::MatNum`: The covariance of the estimated factor returns.

# Related

  - [`attribution_standard_errors`](@ref)
  - [`cross_sectional_rank`](@ref)
  - [`PseudoInverseFallback`](@ref)
"""
function attribution_sandwich(Bt::MatNum, q::VecNum, s2::VecNum,
                              keep::AbstractVector{<:Integer})
    B = view(Bt, :, keep)
    G = transpose(B) * LinearAlgebra.Diagonal(q) * B
    S = transpose(B) * LinearAlgebra.Diagonal(q .^ 2 .* s2) * B
    if cross_sectional_rank(G) == size(G, 2)
        return transpose(G \ transpose(G \ S))
    end
    Gi = LinearAlgebra.pinv(Matrix(G))
    return Gi * S * Gi
end
"""
    attribution_expand_errors(fcb::Nothing, V::MatNum, keep, nr::Integer, t::Integer)
    attribution_expand_errors(fcb::FactorFamilyBasis, V::MatNum, keep, nr::Integer,
                              t::Integer)

Return the sandwich covariance of one observation on the raw factor axis.

The sandwich runs over the factors the regression estimates, so the answer is first scattered back into the full regression axis, and then, under a family re-basis, mapped from the reduced axis onto the raw one.

# Arguments

  - `fcb`: The family re-basis, or `nothing`.
  - `V`: The covariance over the estimated factors.
  - `keep`: The factors the regression estimates.
  - `nr`: The number of columns of the regression basis.
  - `t`: The observation.

# Returns

  - `Vf::MatNum`: The covariance on the raw factor axis.

# Related

  - [`attribution_standard_errors`](@ref)
  - [`expand_factor_covariance`](@ref)
"""
function attribution_expand_errors(::Nothing, V::MatNum, keep, nr::Integer, ::Integer)
    return attribution_scatter(V, keep, nr)
end
function attribution_expand_errors(fcb::FactorFamilyBasis, V::MatNum, keep, nr::Integer,
                                   t::Integer)
    return expand_factor_covariance(fcb, attribution_scatter(V, keep, nr), t)
end
"""
    attribution_scatter(V::MatNum, keep, nr::Integer)

Return a covariance over a subset of factors, placed into a zero matrix of the full axis.

A factor the regression does not estimate carries no estimation uncertainty, so its row and its column are zero rather than absent.

# Arguments

  - `V`: The covariance over the estimated factors.
  - `keep`: The factors the regression estimates.
  - `nr`: The number of factors of the full axis.

# Returns

  - `S::MatNum`: The covariance on the full axis.

# Related

  - [`attribution_expand_errors`](@ref)
"""
function attribution_scatter(V::MatNum, keep, nr::Integer)
    S = zeros(eltype(V), nr, nr)
    for b in eachindex(keep), a in eachindex(keep)
        S[keep[a], keep[b]] = V[a, b]
    end
    return S
end
"""
    attribution_family_errors(fam::Nothing, g::MatNum, Vf::AbstractVector{<:MatNum},
                              s1::Number, T::Integer)
    attribution_family_errors(fam::VecStr, g::MatNum, Vf::AbstractVector{<:MatNum},
                              s1::Number, T::Integer)

Return the standard error of each family's mean return contribution.

A family's contribution sums the contributions of its factors, so its error reads the full covariance block of the family rather than the diagonal alone. The currency family reports `NaN`, because its rows carry no regression-estimation uncertainty.

# Mathematical definition

```math
\\begin{align}
\\mathrm{SE}_{\\mathcal{F}} &= \\frac{p}{T} \\sqrt{\\sum_{t=1}^{T} \\boldsymbol{g}_{t,\\mathcal{F}}^{\\intercal} \\mathbf{V}_{t,\\mathcal{F}\\mathcal{F}}\\, \\boldsymbol{g}_{t,\\mathcal{F}}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{SE}_{\\mathcal{F}}``: Standard error of the mean return contribution of the family.
  - ``\\boldsymbol{g}_{t,\\mathcal{F}}``, ``\\mathbf{V}_{t,\\mathcal{F}\\mathcal{F}}``: The entries of ``\\boldsymbol{g}_{t}``, and the block of ``\\mathbf{V}_{t}``, that the factors of the family index.
  - $(math_dict[:F_fam_att])
  - $(math_dict[:g_t_att])
  - $(math_dict[:V_t_att])
  - $(math_dict[:p_ppy])
  - $(math_dict[:T])

# Arguments

  - `fam`: The family label of each raw factor, or `nothing`.
  - `g`: The per-observation portfolio exposure on the raw axis.
  - `Vf`: The sandwich covariance of each observation, on the raw axis.
  - `s1`: The factor a mean takes under the annualisation.
  - `T`: The number of aligned observations.

# Returns

  - `family::Option{<:VecNum}`: The standard error of each family, or `nothing`.

# Related

  - [`attribution_standard_errors`](@ref)
  - [`attribution_family_index`](@ref)
"""
function attribution_family_errors(::Nothing, ::MatNum, ::AbstractVector{<:MatNum},
                                   ::Number, ::Integer)::Nothing
    return nothing
end
function attribution_family_errors(fam::VecStr, g::MatNum, Vf::AbstractVector{<:MatNum},
                                   s1::Number, T::Integer)
    fi = attribution_family_index(fam)
    se(v) = s1 * sqrt(max(zero(v), v)) / T
    out = [se(sum(LinearAlgebra.dot(view(g, t, i), view(Vf[t], i, i), view(g, t, i))
                  for t in 1:T)) for i in fi.idx]
    for j in eachindex(out)
        if fi.labels[j] == ATTRIBUTION_CURRENCY_FAMILY
            out[j] = oftype(out[j], NaN)
        end
    end
    return out
end
"""
    realised_attribution_assets(assets::Bool, W::VecNum_MatNum, al::NamedTuple,
                                sysr::MatNum, retc::VecNum, total_vol::Number,
                                sc::NamedTuple)

Return the asset axis and the asset-by-factor matrices of a realised attribution.

Each asset's systematic and idiosyncratic contributions are the covariances of its own weighted series with the portfolio, so the rows sum to their components exactly. The axis decomposes the model, so `vol_contrib` is the two parts together, and the rows sum to the systematic and idiosyncratic components together and not to the total. The remainder belongs to the portfolio and has no split over the assets. The asset-by-factor matrices split the systematic row of each asset over the factors.

# Mathematical definition

The model return of asset ``i`` is the sum of its systematic and idiosyncratic returns.

```math
\\begin{align}
a_{ti} &= s_{ti} + \\varepsilon_{ti}\\,.
\\end{align}
```

The systematic row of asset ``i`` is ``\\mathrm{VC}`` and ``\\mathrm{MC}`` of ``w_{\\cdot i} s_{\\cdot i}``, and the idiosyncratic row is the same two numbers of ``w_{\\cdot i} \\varepsilon_{\\cdot i}``. The total row is ``\\mathrm{VC}``, ``\\mathrm{PV}`` and ``\\mathrm{MC}`` of ``w_{\\cdot i} a_{\\cdot i}``. The weighted series of all the assets sum to ``s_{t}`` and ``e_{t}``, so the rows sum to the two components. The standalone numbers are those of the model return, which holds no per-observation intercept.

```math
\\begin{align}
\\mathrm{vol}_{i} &= \\sqrt{p}\\, \\operatorname{sd}(a_{\\cdot i})\\,, \\\\
\\mu_{i} &= \\frac{p}{T} \\sum_{t=1}^{T} a_{ti}\\,, \\\\
\\rho_{i} &= \\frac{\\operatorname{cov}(a_{\\cdot i}, r)}{\\operatorname{sd}(a_{\\cdot i})\\, \\sigma_{P}}\\,.
\\end{align}
```

Where:

  - ``\\varepsilon_{ti}``, ``a_{ti}``: Idiosyncratic and model return of asset ``i`` at observation ``t``.
  - $(math_dict[:s_ti_att])
  - $(math_dict[:s_e_t_att])
  - ``\\mathrm{vol}_{i}``, ``\\mu_{i}``, ``\\rho_{i}``: Standalone volatility, mean return and correlation with the portfolio of asset ``i``.
  - $(math_dict[:B_t_att])
  - $(math_dict[:f_t_att])
  - $(math_dict[:eps_t_att])
  - $(math_dict[:w_t_att])
  - $(math_dict[:r_t_att])
  - $(math_dict[:sigma_P_att])
  - $(math_dict[:VC_att])
  - $(math_dict[:cov_sd_att])
  - $(math_dict[:p_ppy])
  - $(math_dict[:T])

# Arguments

  - `assets`: Whether to compute the two answers at all.
  - `W`: The constant weights, or the weight history.
  - `al`: The aligned factor model history.
  - `sysr`: The per-observation systematic return of each asset.
  - `retc`: The centred portfolio return series.
  - `total_vol`: The portfolio volatility.
  - `sc`: The two annualisation factors.

# Returns

  - `abd::Option{<:AssetAttributionBreakdown}`: The asset axis, or `nothing`.
  - `afc::Option{<:AssetFactorContribution}`: The asset-by-factor matrices, or `nothing`.

# Related

  - [`realised_attribution`](@ref)
  - [`AssetAttributionBreakdown`](@ref)
  - [`AssetFactorContribution`](@ref)
"""
function realised_attribution_assets(assets::Bool, W::VecNum_MatNum, al::NamedTuple,
                                     sysr::MatNum, retc::VecNum, total_vol::Number,
                                     sc::NamedTuple)
    if !assets
        return nothing, nothing
    end
    eps, f = al.eps, al.f
    T, N = size(eps)
    K = size(f, 2)
    ar = sysr .+ eps
    Tf = promote_type(eltype(W), eltype(sysr), eltype(eps))
    syspnl = Matrix{Tf}(undef, T, N)
    idiopnl = Matrix{Tf}(undef, T, N)
    for t in 1:T
        wt = attribution_weights(W, t)
        syspnl[t, :] = wt .* view(sysr, t, :)
        idiopnl[t, :] = wt .* view(eps, t, :)
    end
    svc = [attribution_cov(view(syspnl, :, i), retc) for i in 1:N] ./ total_vol
    ivc = [attribution_cov(view(idiopnl, :, i), retc) for i in 1:N] ./ total_vol
    vc = svc .+ ivc
    vol = vec(std(ar; dims = 1))
    cvp = [attribution_cov(view(ar, :, i), retc) for i in 1:N]
    weight, weight_std = attribution_weight_moments(W, N)
    abd = AssetAttributionBreakdown(weight, weight_std, svc .* sc.s2,
                                    vec(mean(syspnl; dims = 1)) .* sc.s1, ivc .* sc.s2,
                                    vec(mean(idiopnl; dims = 1)) .* sc.s1, vol .* sc.s2,
                                    [attribution_safe_corr(cvp[i], vol[i], total_vol)
                                     for i in 1:N], vc .* sc.s2, vc ./ total_vol,
                                    vec(mean(ar; dims = 1)) .* sc.s1,
                                    vec(mean(syspnl .+ idiopnl; dims = 1)) .* sc.s1)
    afc = realised_attribution_asset_factor(W, al, retc, total_vol, sc, T, N, K)
    return abd, afc
end
"""
    realised_attribution_asset_factor(W::VecNum_MatNum, al::NamedTuple, retc::VecNum,
                                      total_vol::Number, sc::NamedTuple, T::Integer,
                                      N::Integer, K::Integer) -> AssetFactorContribution

Return the asset-by-factor contributions of a realised attribution.

Each pair of an asset and a factor is one term of the systematic return, so a row of the matrices sums to the asset's systematic row and a column sums to the factor's row.

# Mathematical definition

```math
\\begin{align}
c^{(ik)}_{t} &= w_{ti}\\, B_{t,ik}\\, f_{tk}\\,, \\\\
\\mathrm{VC}_{ik} &= \\mathrm{VC}(c^{(ik)})\\,, \\\\
\\mathrm{MC}_{ik} &= \\mathrm{MC}(c^{(ik)})\\,.
\\end{align}
```

``\\sum_{k} c^{(ik)}_{t} = w_{ti} s_{ti}`` and ``\\sum_{i} c^{(ik)}_{t} = g_{tk} f_{tk}``, and both numbers are linear in the series.

Where:

  - ``c^{(ik)}_{t}``: Return of the pair of asset ``i`` and factor ``k`` at observation ``t``.
  - ``B_{t,ik}``: Entry ``(i, k)`` of ``\\mathbf{B}_{t}``.
  - $(math_dict[:s_ti_att])
  - ``\\mathrm{VC}_{ik}``, ``\\mathrm{MC}_{ik}``: Volatility contribution and mean return contribution of the pair.
  - $(math_dict[:B_t_att])
  - $(math_dict[:w_t_att])
  - $(math_dict[:g_t_att])
  - $(math_dict[:f_t_att])
  - $(math_dict[:VC_att])

# Arguments

  - `W`: The constant weights, or the weight history.
  - `al`: The aligned factor model history.
  - `retc`: The centred portfolio return series.
  - `total_vol`: The portfolio volatility.
  - `sc`: The two annualisation factors.
  - `T`: The number of aligned observations.
  - `N`: The number of assets.
  - `K`: The number of raw factors.

# Returns

  - `afc::AssetFactorContribution`: The two asset-by-factor matrices.

# Related

  - [`realised_attribution_assets`](@ref)
  - [`AssetFactorContribution`](@ref)
"""
function realised_attribution_asset_factor(W::VecNum_MatNum, al::NamedTuple, retc::VecNum,
                                           total_vol::Number, sc::NamedTuple, T::Integer,
                                           N::Integer, K::Integer)
    f = al.f
    Tf = promote_type(eltype(W), eltype(al.B), eltype(f), eltype(retc))
    vc = Matrix{Tf}(undef, N, K)
    mc = Matrix{Tf}(undef, N, K)
    pnl = Vector{Tf}(undef, T)
    for k in 1:K, i in 1:N
        for t in 1:T
            pnl[t] = attribution_weights(W, t)[i] *
                     attribution_slice(al.B, t)[i, k] *
                     f[t, k]
        end
        mc[i, k] = mean(pnl) * sc.s1
        vc[i, k] = attribution_cov(pnl, retc) / total_vol * sc.s2
    end
    return AssetFactorContribution(vc, mc)
end
"""
    attribution_rolling(W::VecNum_MatNum, ret::VecNum, al::NamedTuple, fam, assets::Bool,
                        se::Bool, ppy::Number, window::Integer, step::Integer)
        -> Vector{<:FactorAttributionResult}

Roll a realised attribution over the windows of an aligned history.

The alignment runs once over the whole history, and each window is a slice of the aligned axis, so a window carries `window` effective observations and spends the exposure lag once rather than once per window.

# Mathematical definition

```math
\\begin{align}
\\mathcal{W}_{j} &= \\{\\tau_{j} - m + 1, \\ldots, \\tau_{j}\\}\\,, \\\\
\\tau_{j} &= m + (j - 1)\\, h\\,, \\quad j = 1, \\ldots, \\left\\lfloor \\frac{T - m}{h} \\right\\rfloor + 1\\,.
\\end{align}
```

Attribution ``j`` is the realised attribution over the observations of ``\\mathcal{W}_{j}``. Two windows that share observations are not independent.

Where:

  - ``\\mathcal{W}_{j}``, ``\\tau_{j}``: Observations of window ``j``, and its last observation.
  - ``m``: Size of the window, `window`.
  - ``h``: Stride between two consecutive windows, `step`.
  - $(math_dict[:T])

# Arguments

  - `W`: The constant weights, or the weight history.
  - `ret`: The net portfolio return series, already aligned.
  - `al`: The aligned factor model history.
  - `fam`: The family label of each raw factor, or `nothing`.
  - `assets`: Whether to fill the asset axis and the asset-by-factor matrices.
  - `se`: Whether to fill the standard errors of the mean return contributions.
  - `ppy`: Periods per year the numbers are scaled to.
  - `window`: Size of the rolling window, in observations.
  - `step`: Stride between two consecutive windows.

# Validation

  - `2 <= window <= T`, else a `DomainError` is raised. A window of one observation has no sample volatility.
  - `step >= 1`, else a `DomainError` is raised.

# Returns

  - `fas::Vector{<:FactorAttributionResult}`: One attribution per window.

# Related

  - [`factor_attribution`](@ref)
  - [`realised_attribution`](@ref)
  - [`rolling_window_measure`](@ref)
"""
function attribution_rolling(W::VecNum_MatNum, ret::VecNum, al::NamedTuple,
                             fam::Option{<:VecStr}, assets::Bool, se::Bool, ppy::Number,
                             window::Integer, step::Integer)
    T = length(ret)
    @argcheck(2 <= window <= T,
              DomainError(window,
                          "window must be in 2:$(T), where $(T) is the number of aligned observations; a window of one observation has no sample volatility to attribute. Got window => $window"))
    @argcheck(step >= one(step), DomainError(step, "step must be >= 1"))
    return [realised_attribution(attribution_window_weights(W, (t - window + 1):t),
                                 view(ret, (t - window + 1):t),
                                 attribution_window(al, (t - window + 1):t), fam, assets,
                                 se, ppy) for t in window:step:T]
end
"""
    attribution_finite_series(ret::VecNum)

Refuse a portfolio return series that carries a non-finite value, naming the observations.

The methods that form the series themselves route a non-finite return through [`attribution_net_returns`](@ref), so a `NaN` that reaches here came in the caller's own series or in a fold's. It is refused before the alignment, so the observations named are the caller's and not the aligned window's, and the refusal names the cause rather than the undefined volatility that the value would give.

# Arguments

  - `ret`: The net portfolio return series.

# Validation

  - `ret` is finite throughout, else an `IsNonFiniteError` naming the observations is raised.

# Returns

  - Nothing is returned.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_net_returns`](@ref)
  - [`attribution_realised_entry`](@ref)
"""
function attribution_finite_series(ret::VecNum)::Nothing
    nonfinite = findall(!isfinite, ret)
    @argcheck(isempty(nonfinite),
              IsNonFiniteError("the portfolio return series carries a non-finite value at observations $(nonfinite), so no attribution over it exists. A holding earns a return at every observation it is held: form the series over finite returns, or drop the observations."))
    return nothing
end
"""
    attribution_realised_entry(W::VecNum_MatNum, pr::AbstractPriorResult, ret::VecNum;
                               assets::Bool = false, se::Bool = false, ppy::Number = 1,
                               strict::Bool = false)
        -> FactorAttributionResult

Align a factor model block against a realised return series and decompose it.

Every realised method of [`factor_attribution`](@ref) that is not rolling forms its net return series and its weights, then calls this function.

# Algorithm

 1. Read the factor model block `rr` off `pr`.
 2. Report a holding in a non-investable asset through [`attribution_investable_diagnostic`](@ref), which warns, or raises under `strict`.
 3. Refuse a non-finite entry of `ret` through [`attribution_finite_series`](@ref).
 4. Align the block against the `length(ret)` observations of the caller, giving `al`.
 5. Decompose the rows `al.rows` of `ret`, and of `W` when it is a history, with [`realised_attribution`](@ref).

# Arguments

  - `W`: The constant weights, or the weight history.
  - `pr`: Prior result carrying the factor model block.
  - `ret`: The net portfolio return series.
  - `assets`: Whether to fill the asset axis and the asset-by-factor matrices.
  - `se`: Whether to fill the standard errors of the mean return contributions.
  - `ppy`: Periods per year the numbers are scaled to.
  - `strict`: Whether a holding in a non-investable asset raises rather than warns.

# Validation

  - Every weight at a non-investable asset is zero, else a warning names the assets, or an `ArgumentError` names them under `strict`.
  - `ret` is finite throughout, else an `IsNonFiniteError` naming the observations is raised.
  - `ret` carries at least as many observations as the block, and the block more than its exposure lag, else a `DimensionMismatch` is raised.
  - The portfolio volatility is positive, and `ppy > 0`, else a `DomainError` is raised.

# Returns

  - `fa::FactorAttributionResult`: The attribution of the series.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_investable_diagnostic`](@ref)
  - [`attribution_align`](@ref)
  - [`realised_attribution`](@ref)
"""
function attribution_realised_entry(W::VecNum_MatNum, pr::AbstractPriorResult, ret::VecNum;
                                    assets::Bool = false, se::Bool = false, ppy::Number = 1,
                                    strict::Bool = false)::FactorAttributionResult
    rr = attribution_prior_block(pr).rr
    attribution_investable_diagnostic(W, pr, strict)
    attribution_finite_series(ret)
    al = attribution_align(rr, pr, length(ret))
    return realised_attribution(attribution_window_weights(W, al.rows), view(ret, al.rows),
                                al, attribution_families(rr), assets, se, ppy)
end
"""
    attribution_rolling_entry(W::VecNum_MatNum, pr::AbstractPriorResult, ret::VecNum,
                              window::Integer; step::Integer = 1, assets::Bool = false,
                              se::Bool = false, ppy::Number = 1, strict::Bool = false)
        -> Vector{<:FactorAttributionResult}

Align a factor model block against a realised return series and roll the decomposition.

Every rolling method of [`factor_attribution`](@ref) forms its net return series and its weights, then calls this function.

# Algorithm

 1. Read the factor model block `rr` off `pr`.
 2. Report a holding in a non-investable asset through [`attribution_investable_diagnostic`](@ref), which warns, or raises under `strict`.
 3. Refuse a non-finite entry of `ret` through [`attribution_finite_series`](@ref).
 4. Align the block against the `length(ret)` observations of the caller, giving `al`.
 5. Roll the decomposition over the windows of the rows `al.rows` with [`attribution_rolling`](@ref), which checks `window` and `step` against the aligned length.

# Arguments

  - `W`: The constant weights, or the weight history.
  - `pr`: Prior result carrying the factor model block.
  - `ret`: The net portfolio return series.
  - `window`: Size of the rolling window, in observations.
  - `step`: Stride between two consecutive windows.
  - `assets`: Whether to fill the asset axis and the asset-by-factor matrices.
  - `se`: Whether to fill the standard errors of the mean return contributions.
  - `ppy`: Periods per year the numbers are scaled to.
  - `strict`: Whether a holding in a non-investable asset raises rather than warns.

# Validation

  - Every weight at a non-investable asset is zero, else a warning names the assets, or an `ArgumentError` names them under `strict`.
  - `ret` is finite throughout, else an `IsNonFiniteError` naming the observations is raised.
  - `ret` carries at least as many observations as the block, and the block more than its exposure lag, else a `DimensionMismatch` is raised.
  - The portfolio volatility is positive, and `ppy > 0`, else a `DomainError` is raised.
  - `2 <= window <= T` over the aligned observations, else a `DomainError` is raised.
  - `step >= 1`, else a `DomainError` is raised.

# Returns

  - `fas::Vector{<:FactorAttributionResult}`: One attribution per window.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_investable_diagnostic`](@ref)
  - [`attribution_rolling`](@ref)
"""
function attribution_rolling_entry(W::VecNum_MatNum, pr::AbstractPriorResult, ret::VecNum,
                                   window::Integer; step::Integer = 1, assets::Bool = false,
                                   se::Bool = false, ppy::Number = 1, strict::Bool = false)
    rr = attribution_prior_block(pr).rr
    attribution_investable_diagnostic(W, pr, strict)
    attribution_finite_series(ret)
    al = attribution_align(rr, pr, length(ret))
    return attribution_rolling(attribution_window_weights(W, al.rows), ret[al.rows], al,
                               attribution_families(rr), assets, se, ppy, window, step)
end
"""
    attribution_net_returns(w::VecNum, X::MatNum, fees::Option{<:Fees}, strict::Bool)

Return the net portfolio return series over the finite entries of the asset returns.

A point-in-time panel carries a `NaN` at every `(observation, asset)` pair where the asset is inactive: before it lists, after it delists, and at a non-investable asset's whole column. `0 * NaN` is `NaN`, so a zero weight does not remove a `NaN` from the product `X * w`. The series is therefore formed over the finite entries.

A pair with a zero weight contributes nothing whatever it holds, and no message names it. A pair with a non-zero weight and a non-finite return is a holding with no return to earn, and it takes the library's strictness policy through [`strict_diagnostic`](@ref). By default a warning names the observations and the assets, and the pair contributes zero. Under `strict`, an `ArgumentError` names them instead. Under a walk-forward the held pairs after a delisting carry a zero weight already, so the default is silent there.

# Mathematical definition

```math
\\begin{align}
\\tilde{x}_{ti} &= \\begin{cases} x_{ti} & x_{ti} \\text{ finite}\\,, \\\\ 0 & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

The series is the net series [`calc_net_returns`](@ref) forms from ``\\tilde{\\mathbf{X}}`` and ``\\boldsymbol{w}``. A panel with no non-finite entry gives ``\\tilde{\\mathbf{X}} = \\mathbf{X}``.

Where:

  - ``x_{ti}``, ``\\tilde{x}_{ti}``: Return of asset ``i`` at observation ``t``, and its finite part, the entries of ``\\mathbf{X}`` and ``\\tilde{\\mathbf{X}}``.
  - $(math_dict[:w_port])

# Arguments

  - `w`: Portfolio weights.
  - `X`: Asset returns, `observations × assets`.
  - `fees`: Fees the net series is formed against, or `nothing`.
  - `strict`: Whether a non-finite return at a held pair raises rather than warns.

# Validation

  - Every held pair of `X` is finite, else a warning naming the pairs is emitted, or an `ArgumentError` naming them is raised under `strict`.

# Returns

  - `ret::VecNum`: The net portfolio return series, one entry per observation.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_investable_diagnostic`](@ref)
  - [`calc_net_returns`](@ref)
  - [`strict_diagnostic`](@ref)
"""
function attribution_net_returns(w::VecNum, X::MatNum, fees::Option{<:Fees}, strict::Bool)
    if all(isfinite, X)
        return calc_net_returns(w, X, fees)
    end
    held = held_gap_pairs(w, X)
    if !isempty(held)
        assets = unique(last.(held))
        strict_diagnostic("a factor attribution cannot decompose a holding that earns no return. Assets $(assets) carry a non-finite return at $(length(held)) held (observation, asset) pair(s), the first at observation $(first(held)[1]). Those pairs contribute zero to the net series, so the total understates the portfolio by whatever they earned. Pass `strict = true` to refuse instead, zero the weights over the observations the asset is inactive, or pass a weight history.",
                          strict)
    end
    Y = attribution_finite(X)
    return calc_net_returns(w, Y, fees)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, X::MatNum,
                            fees::Option{<:Fees} = nothing; strict::Bool = false,
                            kwargs...)::FactorAttributionResult
    return attribution_realised_entry(w, pr, attribution_net_returns(w, X, fees, strict);
                                      strict = strict, kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, X::MatNum,
                            fees::Option{<:Fees}, window::Integer; strict::Bool = false,
                            kwargs...)
    return attribution_rolling_entry(w, pr, attribution_net_returns(w, X, fees, strict),
                                     window; strict = strict, kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, X::MatNum, window::Integer;
                            kwargs...)
    return factor_attribution(w, pr, X, nothing, window; kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, rd::ReturnsResult,
                            fees::Option{<:Fees} = nothing;
                            kwargs...)::FactorAttributionResult
    return factor_attribution(w, pr, rd.X, fees; kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, rd::ReturnsResult,
                            fees::Option{<:Fees}, window::Integer; kwargs...)
    return factor_attribution(w, pr, rd.X, fees, window; kwargs...)
end
function factor_attribution(w::VecNum, pr::AbstractPriorResult, rd::ReturnsResult,
                            window::Integer; kwargs...)
    return factor_attribution(w, pr, rd.X, nothing, window; kwargs...)
end
function factor_attribution(res::OptimisationResult, pr::Option{<:Pr_RR}, rd::ReturnsResult;
                            kwargs...)::FactorAttributionResult
    # The caller's `rd` is on the universe of `res.w`, as a caller's `pr` is, so its
    # matrix takes the same view the prior and the weights take.
    imsk, w, pr, fees = result_investable_view(res, pr)
    return factor_attribution(w, pr, investable_weights_view(imsk, rd.X), fees; kwargs...)
end
function factor_attribution(res::OptimisationResult, pr::Option{<:Pr_RR}, rd::ReturnsResult,
                            window::Integer; kwargs...)
    imsk, w, pr, fees = result_investable_view(res, pr)
    return factor_attribution(w, pr, investable_weights_view(imsk, rd.X), fees, window;
                              kwargs...)
end
function factor_attribution(W::MatNum, pr::AbstractPriorResult, ret::VecNum;
                            kwargs...)::FactorAttributionResult
    return attribution_realised_entry(W, pr, ret; kwargs...)
end
function factor_attribution(W::MatNum, pr::AbstractPriorResult, ret::VecNum,
                            window::Integer; kwargs...)
    return attribution_rolling_entry(W, pr, ret, window; kwargs...)
end
function factor_attribution(pred::MultiPeriodPredictionResult, pr::AbstractPriorResult;
                            kwargs...)::FactorAttributionResult
    W, ret = attribution_prediction_history(pred)
    return attribution_realised_entry(W, pr, ret; kwargs...)
end
function factor_attribution(pred::MultiPeriodPredictionResult, pr::AbstractPriorResult,
                            window::Integer; kwargs...)
    W, ret = attribution_prediction_history(pred)
    return attribution_rolling_entry(W, pr, ret, window; kwargs...)
end
"""
    attribution_prediction_history(pred::MultiPeriodPredictionResult)

Return the weight history and the net return series a cross-validation produced.

Each fold holds its own weights and its own net series, so the history is the folds stacked in order. A fold that recorded a Held Weights result carries its drifted path, and a fold that recorded none held its target weights for the whole fold.

# Arguments

  - `pred`: A multi-period prediction result.

# Returns

  - `W::MatNum`: The weight history, `observations × assets`.
  - `ret::VecNum`: The net portfolio return series.

# Related

  - [`factor_attribution`](@ref)
  - [`weight_path`](@ref)
  - [`MultiPeriodPredictionResult`](@ref)
"""
function attribution_prediction_history(pred::MultiPeriodPredictionResult)
    W = reduce(vcat, attribution_fold_weights(p) for p in pred.pred)
    ret = reduce(vcat, attribution_fold_returns(p) for p in pred.pred)
    return W, ret
end
"""
    attribution_fold_weights(pred::PredictionResult)

Return the weight history one fold of a cross-validation held.

A fold that recorded a Held Weights result returns its drifted path, and a fold that recorded none held its target weights over every observation of the fold.

# Arguments

  - `pred`: A single-fold prediction result.

# Returns

  - `W::MatNum`: The fold's weight history, `observations × assets`.

# Related

  - [`attribution_prediction_history`](@ref)
  - [`weight_path`](@ref)
"""
function attribution_fold_weights(pred::PredictionResult)
    return attribution_fold_weights(pred, pred.hw)
end
function attribution_fold_weights(pred::PredictionResult, hw::HeldWeightsResult)
    return weight_path(hw, pred.res.w)
end
function attribution_fold_weights(pred::PredictionResult, ::Nothing)
    return repeat(transpose(pred.res.w), length(attribution_fold_returns(pred)))
end
"""
    attribution_fold_returns(pred::PredictionResult)

Return the net return series one fold of a cross-validation produced.

# Arguments

  - `pred`: A single-fold prediction result.

# Returns

  - `ret::VecNum`: The fold's net portfolio return series.

# Related

  - [`attribution_prediction_history`](@ref)
  - [`PredictionResult`](@ref)
"""
function attribution_fold_returns(pred::PredictionResult)
    X = pred.rd.X
    return isa(X, VecVecNum) ? first(X) : X
end
