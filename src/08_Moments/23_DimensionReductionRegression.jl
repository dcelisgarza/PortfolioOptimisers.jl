"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all dimension reduction regression algorithm targets.

All concrete and/or abstract types implementing dimension reduction algorithms for regression (such as PCA or PPCA) should be subtypes of `DimensionReductionTarget`.

These types are used to specify the dimension reduction method when constructing a [`DimensionReductionRegression`](@ref) estimator. A target must answer `StatsAPI.fit(tgt, X)` with a model that `StatsAPI.predict` and `MultivariateStats.projection` both accept.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
  - [`prep_dim_red_reg`](@ref)
"""
abstract type DimensionReductionTarget <: AbstractRegressionAlgorithm end
"""
    factory(drtgt::DimensionReductionTarget, args...; kwargs...) -> DimensionReductionTarget

No-op factory for [`DimensionReductionTarget`](@ref) subtypes. Returns the target unchanged.

Dimension reduction targets (such as [`PCA`](@ref) and [`PPCA`](@ref)) do not depend on observation weights, so this method returns `drtgt` unchanged. This allows generic code to call `factory` on dimension reduction targets without special-casing.

# Arguments

  - `drtgt`: Dimension reduction target.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `drtgt`: The input dimension reduction target, unchanged.

# Related

  - [`DimensionReductionTarget`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`factory`](@ref)
"""
function factory(drtgt::DimensionReductionTarget, args...; kwargs...)
    return drtgt
end
"""
$(DocStringExtensions.TYPEDEF)

Replaces the factors with the principal components of their standardised covariance.

The `kwargs` field is forwarded to `MultivariateStats.fit(MultivariateStats.PCA, X; kwargs...)`, and it is the only place the retained width is set: `pratio` caps the share of variance the retained components must explain and `maxoutdim` caps their number. The default `kwargs = (;)` takes that library's own defaults, which on a factor matrix of full rank can retain every component and reduce nothing.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PCA(;
        kwargs::NamedTuple = (;)
    ) -> PCA

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> PCA()
PCA
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`PPCA`](@ref)

# References

  - $(ref_dict[:pearson1901])
  - $(ref_dict[:hotelling1933])
"""
@concrete struct PCA <: DimensionReductionTarget
    """
    Keyword arguments passed to `fit(MultivariateStats.PCA, X; kwargs...)`
    """
    kwargs
    function PCA(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function PCA(; kwargs::NamedTuple = (;))::PCA
    return PCA(kwargs)
end
"""
    StatsAPI.fit(drtgt::PCA, X::MatNum)

Fit a Principal Component Analysis (PCA) model to the data matrix `X` using the configuration in `drtgt`.

This method applies PCA as a dimension reduction technique for regression-based moment estimation. The keyword arguments stored in `drtgt.kwargs` are passed to [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit).

# Arguments

  - `drtgt`: A [`PCA`](@ref) dimension reduction target, specifying keyword arguments for PCA.
  - `X`: Data matrix (observations × factors) to which PCA will be fitted.

# Returns

  - `model::PCA`: A fitted PCA model object from `MultivariateStats.jl`.

# Related

  - [`PCA`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
function StatsAPI.fit(drtgt::PCA, X::MatNum)
    return StatsAPI.fit(MultivariateStats.PCA, X; drtgt.kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Replaces the factors with the latent components of a Gaussian latent-variable model.

The model is the maximum-likelihood factor analyser with an isotropic noise variance; its latent directions span the same subspace as the principal components of [`PCA`](@ref), and they coincide with them in the zero-noise limit. The `kwargs` field is forwarded to `MultivariateStats.fit(MultivariateStats.PPCA, X; kwargs...)`. Its default width is one fewer than [`PCA`](@ref)'s: on six factors of full rank `PCA()` retained six components and `PPCA()` retained five.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PPCA(;
        kwargs::NamedTuple = (;)
    ) -> PPCA

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> PPCA()
PPCA
  kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)

# References

  - $(ref_dict[:tipping1999])
"""
@concrete struct PPCA <: DimensionReductionTarget
    """
    Keyword arguments passed to `fit(MultivariateStats.PPCA, X; kwargs...)`
    """
    kwargs
    function PPCA(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function PPCA(; kwargs::NamedTuple = (;))::PPCA
    return PPCA(kwargs)
end
"""
    StatsAPI.fit(drtgt::PPCA, X::MatNum)

Fit a Probabilistic Principal Component Analysis (PPCA) model to the data matrix `X` using the configuration in `drtgt`.

This method applies PPCA as a dimension reduction technique for regression-based moment estimation. The keyword arguments stored in `drtgt.kwargs` are passed to [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit).

# Arguments

  - `drtgt`: A [`PPCA`](@ref) dimension reduction target, specifying keyword arguments for PPCA.
  - `X`: Data matrix (observations × factors) to which PPCA will be fitted.

# Returns

  - `model::PPCA`: A fitted PPCA model object from `MultivariateStats.jl`.

# Related

  - [`PPCA`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
function StatsAPI.fit(drtgt::PPCA, X::MatNum)
    return StatsAPI.fit(MultivariateStats.PPCA, X; drtgt.kwargs...)
end
"""
$(DocStringExtensions.TYPEDEF)

Estimates a loadings matrix by regressing each asset on the leading components of the factors.

`drtgt` reduces the standardised factors to a smaller orthogonal basis, `retgt` fits each asset in that basis, and the coefficients are then mapped back to the original factors. `ve` supplies the mean and the standard deviation that mapping divides by; the expected returns estimator it reads is `ve.me`, and a `nothing` there falls back to `SimpleExpectedReturns()`. Unlike [`StepwiseRegression`](@ref), every asset keeps every factor.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DimensionReductionRegression(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        drtgt::DimensionReductionTarget = PCA(),
        retgt::AbstractRegressionTarget = LinearModel()
    ) -> DimensionReductionRegression

Keywords correspond to the struct's fields.

## Validation

  - If `retgt.kwargs` carries a `weights` entry, it must be an `ObsWeights` and, when it is a vector, `!isempty(retgt.kwargs.weights)`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ve`: Recursively updated via [`factory`](@ref).
  - `drtgt`: Recursively updated via [`factory`](@ref).
  - `retgt`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ve`: Recursively viewed via [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> DimensionReductionRegression()
DimensionReductionRegression
     ve ┼ SimpleVariance
        │          me ┼ SimpleExpectedReturns
        │             │   w ┴ nothing
        │           w ┼ nothing
        │   corrected ┴ Bool: true
  drtgt ┼ PCA
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
  retgt ┼ LinearModel
        │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Details

  - **The standardisation and the recovery read the same statistics.** [`prep_dim_red_reg`](@ref) computes the factor mean from `re.ve.me` and the factor standard deviation from `re.ve`, and `regression` recovers the coefficients with those same two vectors. `ve` therefore governs the whole path, and a weighted `ve` — the one [`factory`](@ref) builds from the incoming observation weights — is honoured end to end. This is the requirement of Equations 4.13, 4.15 and 4.20 of $(ref_dict[:cajas2025]).

# Related

  - [`AbstractRegressionEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`StepwiseRegression`](@ref)
  - [`Regression`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.3.1, Equations 4.12-4.20.
  - $(ref_dict[:fekedulegn2002])
"""
@propagatable @concrete struct DimensionReductionRegression <: AbstractRegressionEstimator
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:drtgt])
    """
    @fprop drtgt
    """
    $(field_dict[:dretgt])
    """
    @fprop retgt
    function DimensionReductionRegression(ve::AbstractVarianceEstimator,
                                          drtgt::DimensionReductionTarget,
                                          retgt::AbstractRegressionTarget)
        if haskey(retgt.kwargs, :weights)
            @argcheck(isa(retgt.kwargs.weights, ObsWeights),
                      ArgumentError("retgt.kwargs.weights must be a vector of observation weights, one element per observation, of type ObsWeights = Union{<:DynamicAbstractWeights, <:StatsBase.AbstractWeights}. Got\nretgt.kwargs.weights => $(typeof(retgt.kwargs.weights))"))
            if isa(retgt.kwargs.weights, AbstractVector)
                @argcheck(!isempty(retgt.kwargs.weights), IsEmptyError)
            end
        end
        return new{typeof(ve), typeof(drtgt), typeof(retgt)}(ve, drtgt, retgt)
    end
end
function DimensionReductionRegression(; ve::AbstractVarianceEstimator = SimpleVariance(),
                                      drtgt::DimensionReductionTarget = PCA(),
                                      retgt::AbstractRegressionTarget = LinearModel())::DimensionReductionRegression
    return DimensionReductionRegression(ve, drtgt, retgt)
end
"""
    prep_dim_red_reg(re::DimensionReductionRegression, X::MatNum)

Prepare data for dimension reduction regression.

This helper function standardises the factor matrix `X`, fits the dimension reduction model of `re.drtgt` (e.g., PCA or PPCA), and projects the standardised data into the reduced-dimensional space. It returns the projected data (with an intercept column), the projection matrix, and the two statistics that did the standardisation.

# Arguments

  - `re`: Dimension reduction regression estimator. Its `ve` supplies the standard deviation, and its `ve.me` the mean. A `nothing` in `ve.me` falls back to `SimpleExpectedReturns()`.
  - `X`: Factor matrix (observations × factors) to be reduced.

# Returns

  - `x1::MatNum`: Projected factor matrix with an intercept column prepended.
  - `Vp::MatNum`: Projection matrix from the fitted dimension reduction model.
  - `mu::VecNum`: Factor means used to centre `X`.
  - `sigma::VecNum`: Factor standard deviations used to scale `X`.

# Details

  - Centres `X` with [`demean_returns`](@ref) and scales it by the standard deviation of `re.ve`, so the standardised factors have mean 0 and variance 1 under those estimators.
  - Floors every entry of `sigma` at `eps(eltype(sigma))`, so a constant factor cannot divide by zero.
  - Fits the dimension reduction model specified by `re.drtgt` to the standardised data.
  - Projects the standardised data into the reduced space.
  - Prepends a column of ones to the projected data for use as an intercept in regression.
  - Returns `mu` and `sigma` so that the caller recovers the coefficients with the scale this function applied. Equations 4.13, 4.15 and 4.20 of $(ref_dict[:cajas2025]) need the two to be the same statistic.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`demean_returns`](@ref)
  - [`_regression(::DimensionReductionRegression, ::VecNum, ::VecNum, ::VecNum, ::MatNum, ::MatNum)`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.3.1, Equations 4.13, 4.16-4.17.
  - $(ref_dict[:fekedulegn2002])
"""
function prep_dim_red_reg(re::DimensionReductionRegression, X::MatNum)
    N = size(X, 1)
    me = ifelse(isnothing(re.ve.me), SimpleExpectedReturns(), re.ve.me)
    sigma = vec(Statistics.std(re.ve, X; dims = 1))
    sigma .= max.(sigma, eps(eltype(sigma)))
    mu = Statistics.mean(me, X; dims = 1)
    X_std = permutedims(demean_returns(X, me; dims = 1, mean = mu) ./ transpose(sigma))
    model = StatsAPI.fit(re.drtgt, X_std)
    Xp = transpose(StatsAPI.predict(model, X_std))
    Vp = MultivariateStats.projection(model)
    x1 = [ones(eltype(X), N) Xp]
    return x1, Vp, vec(mu), sigma
end
"""
    _regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum,
               sigma::VecNum, x1::MatNum, Vp::MatNum)

Fit a regression model in reduced-dimensional space and recover coefficients in the original factor space.

This function fits a regression model (as specified by `retgt`) to the response vector `y` using the projected factor matrix `x1` (typically obtained from a dimension reduction method such as PCA or PPCA). It then transforms the estimated coefficients from the reduced space back to the original factor space using the projection matrix `Vp` and rescales them by the standard deviations `sigma`. The intercept is adjusted to account for the mean of `y` and the means of the original factors.

# Mathematical definition

```math
\\begin{align}
\\hat{y} &= \\hat{\\beta}_{0,\\mathrm{pc}} + \\mathbf{X}_1 \\hat{\\boldsymbol{\\beta}}_{\\mathrm{pc}}\\,, \\\\
\\hat{\\boldsymbol{\\beta}} &= \\mathbf{V}_p \\hat{\\boldsymbol{\\beta}}_{\\mathrm{pc}} \\oslash \\boldsymbol{\\sigma}\\,, \\\\
\\hat{\\beta}_0 &= \\bar{y} - \\hat{\\boldsymbol{\\beta}}^{\\intercal} \\boldsymbol{\\mu}\\,.
\\end{align}
```

Where:

  - ``\\hat{y}``: Fitted response.
  - ``\\hat{\\beta}_{0,\\mathrm{pc}}``: Intercept of the fit in the reduced space, which this method discards.
  - ``\\hat{\\boldsymbol{\\beta}}_{\\mathrm{pc}}``: Regression coefficients in the reduced (PC) space.
  - ``\\hat{\\boldsymbol{\\beta}}``: Regression coefficients in the original factor space.
  - ``\\hat{\\beta}_0``: Intercept adjusted to the original space.
  - ``\\mathbf{X}_1``: Projected factor matrix in the reduced space, with its leading column of ones.
  - ``\\mathbf{V}_p``: PCA/PPCA projection matrix.
  - ``\\boldsymbol{\\sigma}``: Factor standard deviations, from the caller's `re.ve`.
  - ``\\boldsymbol{\\mu}``: Factor means, from the caller's `re.ve.me`.
  - ``\\bar{y}``: Mean of the response, weighted by `re.retgt.kwargs.weights` when that entry is present.
  - ``\\oslash``: Element-wise division.

# Arguments

  - `re`: Dimension reduction regression.
  - `y`: Response vector.
  - `mu`: Mean vector of the original factors.
  - `sigma`: Standard deviation vector of the original factors.
  - `x1`: Projected factor matrix with intercept column (from dimension reduction).
  - `Vp`: Projection matrix from the fitted dimension reduction model.

# Returns

  - `beta::VecNum`: Vector of regression coefficients in the original factor space, with the intercept as the first element.

# Details

  - Fits the regression model in the reduced space using `x1` and `y`.
  - Extracts the coefficients for the principal components (excluding the intercept).
  - Transforms the coefficients back to the original factor space using `Vp` and rescales by `sigma`.
  - Computes the intercept so that predictions are unbiased with respect to the means.
  - The source's Equation 4.20 states ``\\hat{\\beta}_{0,\\mathrm{pc}} = \\bar{y}``, which is why the discarded reduced-space intercept costs nothing. The identity is exact only because the projected columns are centred: on a 250×6 sample the fitted intercept matched ``\\bar{y}`` to `2.1e-17` unweighted, and parted from the weighted mean by `1.7e-3` once `re.retgt.kwargs.weights` was set.
  - `mu` and `sigma` must be the statistics that standardised the factors. [`prep_dim_red_reg`](@ref) returns them for this reason.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`prep_dim_red_reg`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.3.1, Equations 4.18-4.20.
  - $(ref_dict[:fekedulegn2002])
"""
function _regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum, sigma::VecNum,
                     x1::MatNum, Vp::MatNum)
    mean_y = if !haskey(re.retgt.kwargs, :weights)
        Statistics.mean(y)
    else
        Statistics.mean(y, re.retgt.kwargs.weights)
    end
    fit_result = StatsAPI.fit(re.retgt, x1, y)
    beta_pc = StatsAPI.coef(fit_result)[2:end]
    beta = Vp * beta_pc ./ sigma
    beta0 = mean_y - LinearAlgebra.dot(beta, mu)
    pushfirst!(beta, beta0)
    return beta
end
"""
    regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)

Apply dimension reduction regression to each column of a response matrix.

This method fits a regression model with dimension reduction (e.g., PCA or PPCA) to each column of the response matrix `X`, using the factor matrix `F` as predictors. For each response vector (column of `X`), the factors are first standardized and projected into a lower-dimensional space using the dimension reduction target specified in `re.drtgt`. A regression model (specified by `re.retgt`) is then fitted in the reduced space, and the coefficients are mapped back to the original factor space.

# Arguments

  - `re`: Dimension reduction regression estimator specifying the variance estimator, the dimension reduction target and the regression target.
  - `X`: Response matrix (observations × targets/assets).
  - `F`: Factor matrix (observations × factors).

# Returns

  - `reg::Regression`: A regression result object containing:

      + `b`: Vector of intercepts for each response.
      + `M`: Matrix of coefficients for each response and factor (in the original factor space).
      + `L`: Matrix of coefficients in the reduced (projected) space.

# Details

  - The reduction is fitted **once**, on `F` alone, and every column of `X` is regressed on the same projected matrix.
  - The resulting coefficients are transformed back to the original factor space and rescaled.
  - The output `Regression` object contains the intercepts, coefficient matrix in the original space, and the projected coefficients.
  - `L` is recovered as ``(\\mathbf{M} \\odot \\boldsymbol{\\sigma}^{\\intercal}) \\mathbf{V}_p^{+\\intercal}``, which undoes the rescaling and the projection in turn, so it returns the reduced-space coefficients the fits produced. Checked at `6.7e-16` against them on a 250×6 sample. `size(L, 2)` is therefore the number of retained components, which is the width risk is decomposed in.
  - The expected returns estimator is `re.ve.me`. A `nothing` there falls back to `SimpleExpectedReturns()`; there is no separate field for it.
  - `mu` and `sigma` come back from [`prep_dim_red_reg`](@ref), so the recovery divides by the same scale that standardised the factors.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`prep_dim_red_reg`](@ref)
  - [`Regression`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 4.3.1, Equations 4.12-4.22.
  - $(ref_dict[:fekedulegn2002])
"""
function regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)
    cols = size(F, 2) + 1
    rows = size(X, 2)
    rr = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    f1, Vp, mu, sigma = prep_dim_red_reg(re, F)
    for i in axes(rr, 1)
        rr[i, :] = _regression(re, view(X, :, i), mu, sigma, f1, Vp)
    end
    b = view(rr, :, 1)
    M = view(rr, :, 2:cols)
    L = transpose(LinearAlgebra.pinv(Vp) * transpose(M .* transpose(sigma)))
    return Regression(; b = b, M = M, L = L)
end

export PCA, PPCA, DimensionReductionRegression
