"""
$(DocStringExtensions.TYPEDEF)

One row of a factor attribution: the volatility, the volatility contribution, the variance share, the mean return contribution and the correlation of one component of the portfolio return.

A component is the systematic part, the idiosyncratic part, the unattributed remainder or the total. The four instances a [`FactorAttributionResult`](@ref) carries hold the same six numbers, so a reader that tabulates one tabulates all four. Every number is scaled to the `ppy` the Result carries.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AttributionComponent(vol, vol_contrib, pct_var, mu_contrib, corr, mu_se)
        -> AttributionComponent

Arguments correspond to the struct's fields, in the order they are declared. The type is a part of a Result, so [`factor_attribution`](@ref) builds it and a caller reads it; there is no keyword constructor, and the type validates nothing of its own.

# Related

  - [`FactorAttributionResult`](@ref)
  - [`factor_attribution`](@ref)
"""
@concrete struct AttributionComponent
    """
    Volatility of the component's own return series. It is `NaN` on the predicted side of the unattributed remainder, which is a gap between two moments and carries no series.
    """
    vol
    """
    Contribution of the component to the portfolio volatility. The four components sum to the portfolio volatility exactly.
    """
    vol_contrib
    """
    Share of the portfolio variance the component explains. The four components sum to one exactly.
    """
    pct_var
    """
    Contribution of the component to the portfolio mean return. The four components sum to the portfolio mean return exactly.
    """
    mu_contrib
    """
    Correlation of the component's return series with the portfolio return series. It is `NaN` on the predicted side of the unattributed remainder.
    """
    corr
    """
    Standard error of `mu_contrib`, or `nothing`. It is filled on the realised side under `se = true`, and it is `nothing` everywhere else.
    """
    mu_se
end
"""
$(DocStringExtensions.TYPEDEF)

The factor axis or the family axis of a factor attribution, one entry per row of the axis.

The factor axis carries no labels, because the factor names are carried input that the caller already holds. The family axis carries its own labels, because they are derived from the family labels of the block and exist nowhere else. Every number is scaled to the `ppy` the Result carries.

A family is a set of factors, so the four additive fields — `exposure`, `vol_contrib`, `pct_var` and `mu_contrib` — carry the sum of the rows of the family, and `vol`, `corr` and `mu` are `nothing`, because no single standalone volatility, correlation or mean return describes a set.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AttributionBreakdown(labels, exposure, exposure_std, vol, corr, vol_contrib, pct_var,
                         mu, mu_contrib, mu_se) -> AttributionBreakdown

Arguments correspond to the struct's fields, in the order they are declared. The type is a part of a Result, so [`factor_attribution`](@ref) builds it and a caller reads it; there is no keyword constructor, and the type validates nothing of its own.

# Related

  - [`FactorAttributionResult`](@ref)
  - [`factor_attribution`](@ref)
"""
@concrete struct AttributionBreakdown
    """
    Label of each row, or `nothing`. It is `nothing` on the factor axis and the sorted unique family labels on the family axis.
    """
    labels
    """
    Portfolio exposure to each row. It is `M' * w` on the predicted side, and the mean of the per-observation exposure on the realised side.
    """
    exposure
    """
    Standard deviation of the per-observation portfolio exposure to each row, or `nothing`. It is `nothing` on the predicted side, which reads one exposure and no history.
    """
    exposure_std
    """
    Standalone volatility of each factor, or `nothing` on the family axis.
    """
    vol
    """
    Correlation of each factor with the portfolio return, or `nothing` on the family axis.
    """
    corr
    """
    Contribution of each row to the portfolio volatility. The rows sum to the systematic component.
    """
    vol_contrib
    """
    Share of the portfolio variance each row explains. The rows sum to the variance share of the systematic component.
    """
    pct_var
    """
    Mean return of each factor, or `nothing` on the family axis.
    """
    mu
    """
    Contribution of each row to the portfolio mean return. The rows sum to the mean return contribution of the systematic component.
    """
    mu_contrib
    """
    Standard error of `mu_contrib`, one entry per row, or `nothing`. It is filled on the realised side under `se = true`, and a row of a currency family reports `NaN`.
    """
    mu_se
end
"""
$(DocStringExtensions.TYPEDEF)

The asset axis of a factor attribution, one entry per asset.

Each asset carries its weight, its standalone moments and its systematic, idiosyncratic and total contributions, so a reader sees which holding drove a factor row. Every number is scaled to the `ppy` the Result carries.

The axis decomposes the factor model, so the systematic rows sum to the systematic component, the idiosyncratic rows to the idiosyncratic component, and `vol_contrib`, which is the two together, to both of them. It does not reach the total: the difference is the unattributed remainder, which is a property of the portfolio and has no per-asset split.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AssetAttributionBreakdown(weight, weight_std, sys_vol_contrib, sys_mu_contrib,
                              idio_vol_contrib, idio_mu_contrib, vol, corr, vol_contrib,
                              pct_var, mu, mu_contrib) -> AssetAttributionBreakdown

Arguments correspond to the struct's fields, in the order they are declared. The type is a part of a Result, so [`factor_attribution`](@ref) builds it and a caller reads it; there is no keyword constructor, and the type validates nothing of its own.

# Related

  - [`FactorAttributionResult`](@ref)
  - [`factor_attribution`](@ref)
"""
@concrete struct AssetAttributionBreakdown
    """
    Weight of each asset. It is the weight vector on the predicted side and under a constant weight, and the mean of the weight history otherwise.
    """
    weight
    """
    Standard deviation of the weight of each asset, or `nothing`. It is `nothing` wherever the weights are constant, and it is filled from a weight history.
    """
    weight_std
    """
    Contribution of each asset to the systematic part of the portfolio volatility.
    """
    sys_vol_contrib
    """
    Contribution of each asset to the systematic part of the portfolio mean return.
    """
    sys_mu_contrib
    """
    Contribution of each asset to the idiosyncratic part of the portfolio volatility.
    """
    idio_vol_contrib
    """
    Contribution of each asset to the idiosyncratic part of the portfolio mean return.
    """
    idio_mu_contrib
    """
    Standalone volatility of each asset.
    """
    vol
    """
    Correlation of each asset with the portfolio return.
    """
    corr
    """
    Contribution of each asset to the portfolio volatility, the systematic and the idiosyncratic parts together. The rows sum to the systematic and idiosyncratic components together.
    """
    vol_contrib
    """
    Share of the portfolio variance each asset explains. The rows sum to the variance share of the systematic and idiosyncratic components together.
    """
    pct_var
    """
    Mean return of each asset.
    """
    mu
    """
    Contribution of each asset to the portfolio mean return, the systematic and the idiosyncratic parts together. The rows sum to the systematic and idiosyncratic components together.
    """
    mu_contrib
end
"""
$(DocStringExtensions.TYPEDEF)

The asset-by-factor contributions of a factor attribution, two matrices of assets by factors.

The entry `(i, k)` is what asset `i` contributed through factor `k`. Summing a column over the assets gives the factor's row of the factor axis, and summing a row over the factors gives the asset's systematic row of the asset axis. Every number is scaled to the `ppy` the Result carries.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AssetFactorContribution(vol_contrib, mu_contrib) -> AssetFactorContribution

Arguments correspond to the struct's fields, in the order they are declared. The type is a part of a Result, so [`factor_attribution`](@ref) builds it and a caller reads it; there is no keyword constructor, and the type validates nothing of its own.

# Related

  - [`FactorAttributionResult`](@ref)
  - [`factor_attribution`](@ref)
"""
@concrete struct AssetFactorContribution
    """
    Contribution to the portfolio volatility, `assets × factors`.
    """
    vol_contrib
    """
    Contribution to the portfolio mean return, `assets × factors`.
    """
    mu_contrib
end
"""
$(DocStringExtensions.TYPEDEF)

Decomposes a portfolio's volatility and mean return over the factors, the factor families and the assets of a factor model.

`FactorAttributionResult` is what [`factor_attribution`](@ref) returns. It carries the four components, the factor axis, and — when the block names families and when the caller asks for assets — the family axis, the asset axis and the asset-by-factor matrices, so an attribution can be tabulated, plotted, compared across runs or asserted on in a test.

Every field that an axis does not have is `nothing`, so a reader dispatches on `::Nothing` rather than branching: `fmbd` is `nothing` when the block names no family, and `abd` and `afc` are `nothing` unless the caller passed `assets = true`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FactorAttributionResult(sys, idio, unattr, total, fbd, fmbd, abd, afc, realised, ppy)
        -> FactorAttributionResult

Arguments correspond to the struct's fields, in the order they are declared. The type is a Result, so [`factor_attribution`](@ref) builds it and a caller reads it; there is no keyword constructor, and the type validates nothing of its own.

# Related

  - [`factor_attribution`](@ref)
  - [`AttributionComponent`](@ref)
  - [`AttributionBreakdown`](@ref)
  - [`AssetAttributionBreakdown`](@ref)
  - [`AssetFactorContribution`](@ref)
"""
@concrete struct FactorAttributionResult <: AbstractResult
    """
    The systematic component, the part of the portfolio the factors explain.
    """
    sys
    """
    The idiosyncratic component, the part of the portfolio the assets' own residuals explain.
    """
    idio
    """
    The unattributed remainder, the part of the portfolio the factor model does not explain.
    """
    unattr
    """
    The total, the portfolio's own volatility and mean return.
    """
    total
    """
    The factor axis, one row per raw factor.
    """
    fbd
    """
    The family axis, one row per factor family, or `nothing` when the block names no family.
    """
    fmbd
    """
    The asset axis, or `nothing` unless the caller passed `assets = true`.
    """
    abd
    """
    The asset-by-factor contributions, or `nothing` unless the caller passed `assets = true`.
    """
    afc
    """
    Whether the attribution reads a realised history. A realised attribution reports a mean return, and a predicted one reports an expected return.
    """
    realised
    """
    Periods per year the numbers are scaled to. Means and variances scale by `ppy`, volatilities by its square root, and shares and correlations not at all.
    """
    ppy
end
"""
    attribution_idiosyncratic_covariance(rr::AbstractLoadingsRegressionResult)
    attribution_idiosyncratic_covariance(rr::CrossSectionalFactorModel)

Return the idiosyncratic covariance a factor attribution adds to the systematic block.

One of the five reads [`factor_attribution`](@ref) takes off a loadings result. The root refuses, so a loadings result that carries no idiosyncratic block is named rather than silently attributed to zero.

# Arguments

  - `rr`: A loadings regression result.

# Validation

  - The root method always raises an `ArgumentError` naming the type.
  - A [`CrossSectionalFactorModel`](@ref) whose `esigma` is `nothing` raises an `IsNothingError`.

# Returns

  - `esigma::VecNum_MatNum`: The idiosyncratic variances, or the idiosyncratic covariance.

# Related

  - [`factor_attribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`idiosyncratic_variances`](@ref)
"""
function attribution_idiosyncratic_covariance(rr::AbstractLoadingsRegressionResult)
    return throw(ArgumentError("`attribution_idiosyncratic_covariance` is not defined for `$(nameof(typeof(rr)))`. A factor attribution reads the idiosyncratic covariance of the block it decomposes, and every member of `AbstractLoadingsRegressionResult` that a caller attributes must add a method beside its own definition."))
end
function attribution_idiosyncratic_covariance(rr::CrossSectionalFactorModel)
    return assert_attribution_field(rr.esigma, :esigma)
end
"""
    attribution_idiosyncratic_returns(rr::AbstractLoadingsRegressionResult)
    attribution_idiosyncratic_returns(rr::CrossSectionalFactorModel)

Return the idiosyncratic return series a realised factor attribution weights by the portfolio.

One of the five reads [`factor_attribution`](@ref) takes off a loadings result. The root refuses, so a loadings result that keeps no residual history is named rather than attributed to zero.

# Arguments

  - `rr`: A loadings regression result.

# Validation

  - The root method always raises an `ArgumentError` naming the type.
  - A [`CrossSectionalFactorModel`](@ref) whose `csr` is `nothing` raises an `IsNothingError`.

# Returns

  - `eps::MatNum`: The idiosyncratic returns, `observations × assets`.

# Related

  - [`factor_attribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
function attribution_idiosyncratic_returns(rr::AbstractLoadingsRegressionResult)
    return throw(ArgumentError("`attribution_idiosyncratic_returns` is not defined for `$(nameof(typeof(rr)))`. A realised factor attribution reads the idiosyncratic return series of the block it decomposes, and every member of `AbstractLoadingsRegressionResult` that a caller attributes must add a method beside its own definition."))
end
function attribution_idiosyncratic_returns(rr::CrossSectionalFactorModel)
    return assert_attribution_field(rr.csr, :csr).eps
end
"""
    attribution_factor_returns(rr::AbstractLoadingsRegressionResult)
    attribution_factor_returns(rr::CrossSectionalFactorModel)

Return the factor return series a realised factor attribution multiplies by the exposures.

One of the five reads [`factor_attribution`](@ref) takes off a loadings result. The series is on the **raw** factor axis, which is the axis the loadings name, so a family re-basis does not move it.

# Arguments

  - `rr`: A loadings regression result.

# Validation

  - The root method always raises an `ArgumentError` naming the type.
  - A [`CrossSectionalFactorModel`](@ref) whose `csr` is `nothing` raises an `IsNothingError`.

# Returns

  - `f::MatNum`: The factor returns, `observations × factors`.

# Related

  - [`factor_attribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`CrossSectionalRegression`](@ref)
"""
function attribution_factor_returns(rr::AbstractLoadingsRegressionResult)
    return throw(ArgumentError("`attribution_factor_returns` is not defined for `$(nameof(typeof(rr)))`. A realised factor attribution reads the factor return series of the block it decomposes, and every member of `AbstractLoadingsRegressionResult` that a caller attributes must add a method beside its own definition."))
end
function attribution_factor_returns(rr::CrossSectionalFactorModel)
    return assert_attribution_field(rr.csr, :csr).f
end
"""
    attribution_exposures(rr::AbstractLoadingsRegressionResult)
    attribution_exposures(rr::CrossSectionalFactorModel)

Return the exposure history a realised factor attribution reads, one slice per observation.

One of the five reads [`factor_attribution`](@ref) takes off a loadings result. A block whose exposures do not move answers its loadings matrix, and the attribution then reads one static slice.

# Arguments

  - `rr`: A loadings regression result.

# Validation

  - The root method always raises an `ArgumentError` naming the type.

# Returns

  - `Ms::Union{<:MatNum, <:Arr3Num}`: The exposure history `observations × assets × factors`, or the static loadings `assets × factors`.

# Related

  - [`factor_attribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`attribution_lag`](@ref)
"""
function attribution_exposures(rr::AbstractLoadingsRegressionResult)
    return throw(ArgumentError("`attribution_exposures` is not defined for `$(nameof(typeof(rr)))`. A realised factor attribution reads the exposure history of the block it decomposes, and every member of `AbstractLoadingsRegressionResult` that a caller attributes must add a method beside its own definition."))
end
function attribution_exposures(rr::CrossSectionalFactorModel)
    return attribution_exposures(rr.Ms, rr.M)
end
function attribution_exposures(::Nothing, M::MatNum)
    return M
end
function attribution_exposures(Ms::Arr3Num, ::MatNum)
    return Ms
end
"""
    attribution_lag(rr::AbstractLoadingsRegressionResult)
    attribution_lag(rr::CrossSectionalFactorModel)

Return the number of observations by which the exposures lag the returns.

One of the five reads [`factor_attribution`](@ref) takes off a loadings result. The attribution keeps the exposures of observation `t - lag` with the returns of observation `t`, so a block that states no lag answers zero and the two axes line up as they stand.

# Arguments

  - `rr`: A loadings regression result.

# Validation

  - The root method always raises an `ArgumentError` naming the type.

# Returns

  - `lag::Int`: The exposure lag, as a count.

# Related

  - [`factor_attribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`cs_regression_lag`](@ref)
"""
function attribution_lag(rr::AbstractLoadingsRegressionResult)
    return throw(ArgumentError("`attribution_lag` is not defined for `$(nameof(typeof(rr)))`. A realised factor attribution aligns the exposures against the returns by the lag of the block it decomposes, and every member of `AbstractLoadingsRegressionResult` that a caller attributes must add a method beside its own definition."))
end
function attribution_lag(rr::CrossSectionalFactorModel)
    return cs_regression_lag(rr.lag)
end
"""
    attribution_families(rr::AbstractLoadingsRegressionResult)
    attribution_families(rr::CrossSectionalFactorModel)

Return the family label of each raw factor, or `nothing` when the block names none.

The family axis of a [`FactorAttributionResult`](@ref) exists exactly when this read answers a vector, so a block type that carries no family concept answers `nothing` through the root and needs no method of its own.

# Arguments

  - `rr`: A loadings regression result.

# Returns

  - `fam::Option{<:VecStr}`: The family label of each raw factor, or `nothing`.

# Related

  - [`factor_attribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`attribution_family_basis`](@ref)
"""
function attribution_families(::AbstractLoadingsRegressionResult)::Nothing
    return nothing
end
function attribution_families(rr::CrossSectionalFactorModel)
    return rr.fam
end
"""
    attribution_family_basis(rr::AbstractLoadingsRegressionResult)
    attribution_family_basis(rr::CrossSectionalFactorModel)

Return the family re-basis the block's fit was written in, or `nothing`.

The standard errors of a realised attribution are computed in the reduced full-rank basis and mapped back onto the raw axis, because a constrained family makes the raw Gram matrix singular. A block type that constrains no family answers `nothing` through the root, and the sandwich then runs on the raw axis.

# Arguments

  - `rr`: A loadings regression result.

# Returns

  - `fcb::Option{<:AbstractFactorFamilyBasis}`: The family re-basis, or `nothing`.

# Related

  - [`factor_attribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`FactorFamilyBasis`](@ref)
"""
function attribution_family_basis(::AbstractLoadingsRegressionResult)::Nothing
    return nothing
end
function attribution_family_basis(rr::CrossSectionalFactorModel)
    return rr.fcb
end
"""
    attribution_regression_weights(rr::AbstractLoadingsRegressionResult)
    attribution_regression_weights(rr::CrossSectionalFactorModel)

Return the regression weight history the sandwich covariance reads, or `nothing`.

The standard errors of a realised attribution need the weight each asset carried in the fit of each observation. A block type that records none answers `nothing` through the root, and `se = true` then refuses.

# Arguments

  - `rr`: A loadings regression result.

# Returns

  - `rw::Option{<:MatNum}`: The regression weight history `observations × assets`, or `nothing`.

# Related

  - [`factor_attribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`attribution_idiosyncratic_variances`](@ref)
"""
function attribution_regression_weights(::AbstractLoadingsRegressionResult)::Nothing
    return nothing
end
function attribution_regression_weights(rr::CrossSectionalFactorModel)
    return rr.rw
end
"""
    attribution_idiosyncratic_variances(rr::AbstractLoadingsRegressionResult)
    attribution_idiosyncratic_variances(rr::CrossSectionalFactorModel)

Return the idiosyncratic variance history the sandwich covariance reads, or `nothing`.

The standard errors of a realised attribution need the idiosyncratic variance of each asset at each observation. A block type that records none answers `nothing` through the root, and `se = true` then refuses.

# Arguments

  - `rr`: A loadings regression result.

# Returns

  - `vs::Option{<:MatNum}`: The idiosyncratic variance history `observations × assets`, or `nothing`.

# Related

  - [`factor_attribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
  - [`attribution_regression_weights`](@ref)
"""
function attribution_idiosyncratic_variances(::AbstractLoadingsRegressionResult)::Nothing
    return nothing
end
function attribution_idiosyncratic_variances(rr::CrossSectionalFactorModel)
    return rr.vs
end
"""
    assert_attribution_field(x::Nothing, sym::Symbol)
    assert_attribution_field(x, sym::Symbol)

Return an optional field of a factor model block, or raise naming it.

The five reads a factor attribution takes are optional fields of the block, so one refusal names the field a caller must keep rather than letting a `nothing` reach the arithmetic.

# Arguments

  - `x`: The field's value.
  - `sym`: The field's name.

# Validation

  - `x` is not `nothing`, else an `IsNothingError` naming the field is raised.

# Returns

  - `x`: The field's value, unchanged.

# Related

  - [`factor_attribution`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function assert_attribution_field(::Nothing, sym::Symbol)
    return throw(IsNothingError("$(sym) cannot be nothing: a factor attribution reads it off the factor model block it decomposes"))
end
function assert_attribution_field(x, ::Symbol)
    return x
end
"""
    attribution_finite(A::AbstractArray)

Return an array with every non-finite entry replaced by zero, or the array itself.

A Prior Estimator fits on the coverage universe and answers on the full one, so an asset it could not estimate carries `NaN` in its row of `mu`, of `sigma` and of the block. Such an asset carries no weight, and a factor attribution therefore contributes nothing through it, but `0 * NaN` is `NaN` and would poison every sum. The entries are replaced once here rather than guarded at each of the sums.

An array that is finite throughout is returned unchanged and is not copied, which is the whole universe's case.

# Arguments

  - `A`: The array.

# Returns

  - `A::AbstractArray`: The array, with every non-finite entry replaced by zero.

# Related

  - [`factor_attribution`](@ref)
  - [`assert_attribution_investable`](@ref)
  - [`investable_mask`](@ref)
"""
function attribution_finite(A::AbstractArray)
    return all(isfinite, A) ? A : map(x -> isfinite(x) ? x : zero(x), A)
end
"""
    assert_attribution_investable(w::VecNum_MatNum, pr::AbstractPriorResult)

Refuse a portfolio that holds an asset the prior could not estimate.

A non-investable asset carries `NaN` in `mu` and on the diagonal of `sigma`, so no moment of it exists to attribute. A portfolio that holds none of it is decomposed exactly, with a zero row wherever the asset appears; a portfolio that holds some of it has a return the model cannot describe, and the refusal names the assets rather than reporting a decomposition that silently drops them.

# Arguments

  - `w`: The constant weights, or the weight history.
  - `pr`: The prior result.

# Validation

  - Every weight at a non-investable asset is zero, else an `ArgumentError` naming the assets is raised.

# Returns

  - Nothing is returned.

# Related

  - [`factor_attribution`](@ref)
  - [`investable_mask`](@ref)
  - [`attribution_finite`](@ref)
"""
function assert_attribution_investable(w::VecNum_MatNum, pr::AbstractPriorResult)::Nothing
    return assert_attribution_investable(w, investable_mask(pr))
end
function assert_attribution_investable(::VecNum_MatNum, ::Nothing)::Nothing
    return nothing
end
function assert_attribution_investable(w::VecNum, imsk::BitVector)::Nothing
    held = findall(i -> !imsk[i] && !iszero(w[i]), eachindex(imsk))
    @argcheck(isempty(held),
              ArgumentError("a factor attribution cannot decompose a holding in an asset the prior could not estimate. Assets $(held) are not investable, and the portfolio holds them. Reduce the weights to the investable universe, or refit the prior over a history that covers these assets."))
    return nothing
end
function assert_attribution_investable(W::MatNum, imsk::BitVector)::Nothing
    held = findall(i -> !imsk[i] && any(!iszero, view(W, :, i)), eachindex(imsk))
    @argcheck(isempty(held),
              ArgumentError("a factor attribution cannot decompose a holding in an asset the prior could not estimate. Assets $(held) are not investable, and the weight history holds them. Reduce the weights to the investable universe, or refit the prior over a history that covers these assets."))
    return nothing
end
"""
    attribution_prior_block(pr::AbstractPriorResult)

Return the factor model block and the factor distribution a factor attribution decomposes.

A prior result carries the loadings in `rr` and the factor distribution in `fpr`, and its constructor keeps the two together, so one check establishes the whole block. Every wrapping prior forwards both unchanged while it replaces `mu` and `sigma`, which is why the totals anchor on the carrier and the gaps land in the unattributed remainder.

# Arguments

  - `pr`: A prior result.

# Validation

  - `pr.rr` is not `nothing`, else the `IsNothingError` of [`assert_prior_regression`](@ref) is raised.

# Returns

  - `rr::AbstractLoadingsRegressionResult`: The factor model block.
  - `fpr::LowOrderPrior`: The factor distribution.

# Related

  - [`factor_attribution`](@ref)
  - [`assert_prior_regression`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function attribution_prior_block(pr::AbstractPriorResult)
    assert_prior_regression(pr)
    return (; rr = pr.rr, fpr = pr.fpr)
end
"""
    attribution_scale(ppy::Number)

Return the two scale factors an annualisation applies.

Means and variances scale by `ppy`, and volatilities by its square root. Shares and correlations are ratios of two quantities that scale alike, so they are not scaled at all.

# Arguments

  - `ppy`: Periods per year the numbers are scaled to.

# Validation

  - `ppy > 0`, else a `DomainError` is raised.

# Returns

  - `s1::Real`: The factor a mean or a variance takes.
  - `s2::Real`: The factor a volatility takes.

# Related

  - [`factor_attribution`](@ref)
  - [`FactorAttributionResult`](@ref)
"""
function attribution_scale(ppy::Number)
    @argcheck(ppy > zero(ppy), DomainError(ppy, "ppy must be positive"))
    return (; s1 = float(ppy), s2 = sqrt(float(ppy)))
end
"""
    attribution_idiosyncratic_matrix(esigma::VecNum)
    attribution_idiosyncratic_matrix(esigma::MatNum)

Return the idiosyncratic covariance as a matrix the quadratic form reads.

The idiosyncratic block takes two shapes, a vector of variances and a full covariance, so the shape is the dispatch and no caller tests it.

# Arguments

  - `esigma`: The idiosyncratic variances, or the idiosyncratic covariance.

# Returns

  - `D::AbstractMatrix`: The idiosyncratic covariance.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_idiosyncratic_covariance`](@ref)
"""
function attribution_idiosyncratic_matrix(esigma::VecNum)
    return LinearAlgebra.Diagonal(esigma)
end
function attribution_idiosyncratic_matrix(esigma::MatNum)
    return esigma
end
"""
    attribution_safe_corr(cv::Number, s1::Number, s2::Number)

Return a correlation, or `NaN` when the pair of volatilities cannot normalise it.

A factor whose standalone volatility is zero has no correlation with anything, and the quotient would be an infinity or a `NaN` of the arithmetic's own choosing. The verb answers `NaN` so that every such row reads alike.

# Arguments

  - `cv`: The covariance.
  - `s1`: The first volatility.
  - `s2`: The second volatility.

# Returns

  - `rho::Real`: The correlation, or `NaN`.

# Related

  - [`factor_attribution`](@ref)
"""
function attribution_safe_corr(cv::Number, s1::Number, s2::Number)
    d = s1 * s2
    return d > zero(d) ? cv / d : convert(typeof(float(cv / oneunit(d))), NaN)
end
"""
    attribution_family_index(fam::VecStr)

Return the rows of the family axis and the raw factors each of them sums.

The axis is the sorted unique labels, so it is deterministic and independent of the order the factors were built in. Every consumer of the family axis reads the axis from here, so the labels, the sums, the exposure spread and the standard errors are in one order.

# Arguments

  - `fam`: The family label of each raw factor.

# Returns

  - `labels::Vector{String}`: The sorted unique family labels.
  - `idx::Vector{Vector{Int}}`: The raw factors of each family.

# Related

  - [`factor_attribution`](@ref)
  - [`attribution_family_axis`](@ref)
"""
function attribution_family_index(fam::VecStr)
    labels = sort(unique(String[String(f) for f in fam]))
    return (; labels = labels, idx = [findall(f -> String(f) == l, fam) for l in labels])
end
"""
    attribution_family_axis(fam::Nothing, args...)
    attribution_family_axis(fam::VecStr, fbd::AttributionBreakdown, exposure_std, mu_se)

Return the family axis of a factor attribution as a sum of the rows of the factor axis.

A family is a set of factors, so the four additive fields sum over the rows of the family. The axis is the sorted unique labels, which makes it deterministic and independent of the order the factors were built in; a plot that shows the largest families sorts the rows it draws. `vol`, `corr` and `mu` are `nothing`, because no single standalone volatility, correlation or mean return describes a set of factors.

# Arguments

  - `fam`: The family label of each raw factor, or `nothing`.
  - `fbd`: The factor axis.
  - `exposure_std`: The standard deviation of the per-observation family exposure, or `nothing`.
  - `mu_se`: The standard error of the family mean return contribution, or `nothing`.

# Returns

  - `fmbd::Option{<:AttributionBreakdown}`: The family axis, or `nothing` when the block names no family.

# Related

  - [`factor_attribution`](@ref)
  - [`AttributionBreakdown`](@ref)
"""
function attribution_family_axis(::Nothing, args...)::Nothing
    return nothing
end
function attribution_family_axis(fam::VecStr, fbd::AttributionBreakdown,
                                 exposure_std::Option{<:VecNum}, mu_se::Option{<:VecNum})
    fi = attribution_family_index(fam)
    total(v) = [sum(view(v, i)) for i in fi.idx]
    return AttributionBreakdown(fi.labels, total(fbd.exposure), exposure_std, nothing,
                                nothing, total(fbd.vol_contrib), total(fbd.pct_var),
                                nothing, total(fbd.mu_contrib), mu_se)
end
"""
    factor_attribution(w::VecNum, pr::AbstractPriorResult; assets::Bool = false,
                       ppy::Number = 1) -> FactorAttributionResult
    factor_attribution(res::OptimisationResult, pr::Option{<:Pr_RR} = nothing;
                       kwargs...) -> FactorAttributionResult
    factor_attribution(w::VecNum, pr::AbstractPriorResult, X::MatNum,
                       fees::Option{<:Fees} = nothing; assets::Bool = false,
                       se::Bool = false, ppy::Number = 1) -> FactorAttributionResult
    factor_attribution(w::VecNum, pr::AbstractPriorResult, rd::ReturnsResult,
                       fees::Option{<:Fees} = nothing; kwargs...) -> FactorAttributionResult
    factor_attribution(res::OptimisationResult, pr::Option{<:Pr_RR}, rd::ReturnsResult;
                       kwargs...) -> FactorAttributionResult
    factor_attribution(W::MatNum, pr::AbstractPriorResult, ret::VecNum;
                       kwargs...) -> FactorAttributionResult
    factor_attribution(pred::MultiPeriodPredictionResult, pr::AbstractPriorResult;
                       kwargs...) -> FactorAttributionResult
    factor_attribution(args..., window::Integer; step::Integer = 1,
                       kwargs...) -> Vector{<:FactorAttributionResult}

Decompose a portfolio's volatility and mean return over the factors of a factor model.

The verb reads the weights and the factor model block, and returns one [`FactorAttributionResult`](@ref). The **predicted** methods take no return series and decompose the moments the optimiser saw. The **realised** methods take one, and decompose the history the portfolio actually produced. Each realised method has a **rolling** twin that takes a positional `window` and returns one Result per window.

**The predicted totals anchor on the prior result, not on the model.** `pr.mu` and `pr.sigma` are what the optimiser saw and what [`expected_return`](@ref) and [`expected_risk`](@ref) report, so they are the totals. A wrapping prior replaces them while it forwards the block unchanged, so the model no longer reproduces them, and the two gaps `dot(w, pr.mu - M * fpr.mu - b)` and `dot(w, (pr.sigma - M * F * M' - D) * w) / sigma_P` land in the unattributed remainder. The remainder is therefore present on the predicted side too, and it is at rounding level on a plain fit.

**Every source of unexplained return lands in the remainder, and no guard reports it.** On the realised side the identity per observation is `portfolio return = systematic + idiosyncratic + unattributed`, and the remainder holds the per-observation intercept share `b_t * sum(w)`, the fees, the cash, the weight drift inside a period and the exposure lag. A large `pct_var` on the remainder means the model does not explain the portfolio, and the reader draws that conclusion.

**The factor shares disagree with [`factor_risk_contribution`](@ref), and the disagreement is the leakage term.** A risk contribution re-bases the whole portfolio risk into the factor basis and charges the residual leakage to the factors, while this decomposition holds the idiosyncratic part apart as its own component. The two therefore report different factor rows on the same portfolio, and neither is wrong.

# Algorithm

 1. Read the block and the factor distribution off `pr`, and the five series off the block.
 2. Anchor the totals: `pr.mu` and `pr.sigma` on the predicted side, the return series on the realised side.
 3. Decompose the systematic and the idiosyncratic parts, and put every gap into the remainder.
 4. Sum the factor rows by family when the block names families, and over the assets when `assets = true`.
 5. Scale by `ppy`: means and variances by `ppy`, volatilities by its square root.

# Arguments

  - `w`: Portfolio weights.
  - `W`: Portfolio weight history, `observations × assets`.
  - `pr`: Prior result carrying the factor model block.
  - `res`: Optimisation result whose weights and prior the verb extracts.
  - `X`: Asset returns, `observations × assets`.
  - `rd`: Returns result carrying the asset returns.
  - `ret`: Net portfolio return series.
  - `pred`: Multi-period prediction result whose folds give the weight history.
  - `fees`: Fees the net series is formed against.
  - `window`: Size of the rolling window, in observations.
  - `step`: Stride between two consecutive windows.
  - `assets`: Whether to fill the asset axis and the asset-by-factor matrices.
  - `se`: Whether to fill the standard errors of the mean return contributions.
  - `ppy`: Periods per year the numbers are scaled to.

# Validation

  - `pr` carries a factor model block, else the `IsNothingError` of [`assert_prior_regression`](@ref) is raised.
  - `ppy > 0`, else a `DomainError` is raised.
  - The block carries the fields the chosen decomposition reads, else an `IsNothingError` names the field.
  - The portfolio variance is positive on the predicted side, and the portfolio volatility is positive on the realised side, else a `DomainError` is raised.
  - `1 <= window <= T`, else a `DomainError` is raised.
  - `step >= 1`, else a `DomainError` is raised.
  - `se = true` needs the regression weight history and the idiosyncratic variance history, else an `IsNothingError` names the field.

# Returns

  - `fa::FactorAttributionResult`: The attribution, or a vector of them from a rolling method.

# Related

  - [`FactorAttributionResult`](@ref)
  - [`factor_risk_contribution`](@ref)
  - [`plot_attribution_vol_contrib`](@ref)
  - [`plot_attribution_mu_contrib`](@ref)
  - [`plot_attribution_exposure`](@ref)
  - [`plot_attribution_mu_vs_vol`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function factor_attribution(w::VecNum, pr::AbstractPriorResult; assets::Bool = false,
                            ppy::Number = 1)::FactorAttributionResult
    blk = attribution_prior_block(pr)
    rr, fpr = blk.rr, blk.fpr
    assert_attribution_investable(w, pr)
    sc = attribution_scale(ppy)
    M = attribution_finite(rr.M)
    F = fpr.sigma
    mu_f = fpr.mu
    bp = attribution_finite(rr.b)
    sigma = attribution_finite(pr.sigma)
    mu = attribution_finite(pr.mu)
    D = attribution_idiosyncratic_matrix(attribution_finite(attribution_idiosyncratic_covariance(rr)))
    bexp = transpose(M) * w
    Fb = F * bexp
    sys_var = LinearAlgebra.dot(bexp, Fb)
    idio_var = LinearAlgebra.dot(w, D, w)
    total_var = LinearAlgebra.dot(w, sigma, w)
    @argcheck(total_var > zero(total_var),
              DomainError(total_var,
                          "the portfolio variance w' * pr.sigma * w must be positive for an attribution to divide by its square root"))
    sigma_p = sqrt(total_var)
    sys_mu = LinearAlgebra.dot(bexp, mu_f)
    idio_mu = LinearAlgebra.dot(w, bp)
    total_mu = LinearAlgebra.dot(w, mu)
    nan = convert(typeof(sigma_p), NaN)
    sys = AttributionComponent(sqrt(max(sys_var, zero(sys_var))) * sc.s2,
                               sys_var / sigma_p * sc.s2, sys_var / total_var,
                               sys_mu * sc.s1, sqrt(max(sys_var, zero(sys_var))) / sigma_p,
                               nothing)
    idio = AttributionComponent(sqrt(max(idio_var, zero(idio_var))) * sc.s2,
                                idio_var / sigma_p * sc.s2, idio_var / total_var,
                                idio_mu * sc.s1,
                                sqrt(max(idio_var, zero(idio_var))) / sigma_p, nothing)
    gap_var = total_var - sys_var - idio_var
    unattr = AttributionComponent(nan, gap_var / sigma_p * sc.s2, gap_var / total_var,
                                  (total_mu - sys_mu - idio_mu) * sc.s1, nan, nothing)
    total = AttributionComponent(sigma_p * sc.s2, sigma_p * sc.s2, one(total_var),
                                 total_mu * sc.s1, one(total_var), nothing)
    f_vol = sqrt.(max.(LinearAlgebra.diag(F), zero(eltype(F))))
    fbd = AttributionBreakdown(nothing, bexp, nothing, f_vol .* sc.s2,
                               [attribution_safe_corr(Fb[k], f_vol[k], sigma_p)
                                for k in eachindex(Fb)], bexp .* Fb ./ sigma_p .* sc.s2,
                               bexp .* Fb ./ total_var, mu_f .* sc.s1,
                               bexp .* mu_f .* sc.s1, nothing)
    fmbd = attribution_family_axis(attribution_families(rr), fbd, nothing, nothing)
    abd, afc = predicted_attribution_assets(assets, w, M, F, mu_f, D, bp, Fb, sigma_p, sc)
    return FactorAttributionResult(sys, idio, unattr, total, fbd, fmbd, abd, afc, false,
                                   ppy)
end
function factor_attribution(res::OptimisationResult, pr::Option{<:Pr_RR} = nothing;
                            kwargs...)::FactorAttributionResult
    return factor_attribution(res.w, extract_pr(res, pr); kwargs...)
end
"""
    predicted_attribution_assets(assets::Bool, w, M, F, mu_f, D, bp, Fb, sigma_p, sc)

Return the asset axis and the asset-by-factor matrices of a predicted attribution.

The asset axis decomposes the **model**, not the anchors: every row reads `M F M' + D` and `M mu_f + b`, so the systematic rows sum to the systematic component, the idiosyncratic rows to the idiosyncratic component, and `vol_contrib` to the two together. It therefore does not reach the total, and the difference is the unattributed remainder, which is a property of the portfolio and has no per-asset split. The realised asset axis satisfies the same identity, so a reader compares the two sides row by row.

# Arguments

  - `assets`: Whether to compute the two answers at all.
  - `w`: Portfolio weights.
  - `M`: Raw loadings, `assets × factors`.
  - `F`: Factor covariance.
  - `mu_f`: Expected factor returns.
  - `D`: Idiosyncratic covariance.
  - `bp`: Factor-orthogonal expected return.
  - `Fb`: The product of the factor covariance and the portfolio exposure.
  - `sigma_p`: Portfolio volatility.
  - `sc`: The two annualisation factors.

# Returns

  - `abd::Option{<:AssetAttributionBreakdown}`: The asset axis, or `nothing`.
  - `afc::Option{<:AssetFactorContribution}`: The asset-by-factor matrices, or `nothing`.

# Related

  - [`factor_attribution`](@ref)
  - [`AssetAttributionBreakdown`](@ref)
  - [`AssetFactorContribution`](@ref)
"""
function predicted_attribution_assets(assets::Bool, w::VecNum, M::MatNum, F::MatNum,
                                      mu_f::VecNum, D, bp::VecNum, Fb::VecNum,
                                      sigma_p::Number, sc::NamedTuple)
    if !assets
        return nothing, nothing
    end
    sys_cov = M * F * transpose(M)
    full_cov = sys_cov + D
    cov_p = full_cov * w
    vol = sqrt.(max.(LinearAlgebra.diag(full_cov), zero(eltype(sys_cov))))
    vol_contrib = w .* cov_p ./ sigma_p
    sys_mu_contrib = w .* (M * mu_f)
    idio_mu_contrib = w .* bp
    abd = AssetAttributionBreakdown(w, nothing, w .* (sys_cov * w) ./ sigma_p .* sc.s2,
                                    sys_mu_contrib .* sc.s1,
                                    w .* (D * w) ./ sigma_p .* sc.s2,
                                    idio_mu_contrib .* sc.s1, vol .* sc.s2,
                                    [attribution_safe_corr(cov_p[i], vol[i], sigma_p)
                                     for i in eachindex(vol)], vol_contrib .* sc.s2,
                                    vol_contrib ./ sigma_p, (M * mu_f .+ bp) .* sc.s1,
                                    (sys_mu_contrib .+ idio_mu_contrib) .* sc.s1)
    afc = AssetFactorContribution((w * transpose(Fb)) .* M ./ sigma_p .* sc.s2,
                                  (w * transpose(mu_f)) .* M .* sc.s1)
    return abd, afc
end

export AttributionComponent, AttributionBreakdown, AssetAttributionBreakdown,
       AssetFactorContribution, FactorAttributionResult, factor_attribution
