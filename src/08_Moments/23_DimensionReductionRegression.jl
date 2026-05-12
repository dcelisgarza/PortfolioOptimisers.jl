"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all dimension reduction regression algorithm targets in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing dimension reduction algorithms for regression (such as PCA or PPCA) should be subtypes of `DimensionReductionTarget`.

These types are used to specify the dimension reduction method when constructing a [`DimensionReductionRegression`](@ref) estimator.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
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

Principal Component Analysis (PCA) dimension reduction target.

`PCA` is used to specify principal component analysis as the dimension reduction method for regression-based moment estimation. The `kwargs` field stores keyword arguments to be passed to the underlying PCA implementation (e.g., from `MultivariateStats.jl`).

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
"""
@concrete struct PCA <: DimensionReductionTarget
    "Keyword arguments passed to `fit(MultivariateStats.PCA, X; kwargs...)`"
    kwargs
    function PCA(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function PCA(; kwargs::NamedTuple = (;))
    return PCA(kwargs)
end
"""
    StatsAPI.fit(drtgt::PCA, X::MatNum)

Fit a Principal Component Analysis (PCA) model to the data matrix `X` using the configuration in `drtgt`.

This method applies PCA as a dimension reduction technique for regression-based moment estimation. The keyword arguments stored in `drtgt.kwargs` are passed to [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit).

# Arguments

  - `drtgt`: A [`PCA`](@ref) dimension reduction target, specifying keyword arguments for PCA.
  - `X`: Data matrix (observations × features) to which PCA will be fitted.

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

Probabilistic Principal Component Analysis (PPCA) dimension reduction target.

`PPCA` is used to specify probabilistic principal component analysis as the dimension reduction method for regression-based moment estimation. The `kwargs` field stores keyword arguments to be passed to the underlying PPCA implementation (e.g., from `MultivariateStats.jl`).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PPCA(;
        kwargs::NamedTuple = (),
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
"""
@concrete struct PPCA <: DimensionReductionTarget
    "Keyword arguments passed to `fit(MultivariateStats.PPCA, X; kwargs...)`"
    kwargs
    function PPCA(kwargs::NamedTuple)
        return new{typeof(kwargs)}(kwargs)
    end
end
function PPCA(; kwargs::NamedTuple = (;))
    return PPCA(kwargs)
end
"""
    StatsAPI.fit(drtgt::PPCA, X::MatNum)

Fit a Probabilistic Principal Component Analysis (PPCA) model to the data matrix `X` using the configuration in `drtgt`.

This method applies PPCA as a dimension reduction technique for regression-based moment estimation. The keyword arguments stored in `drtgt.kwargs` are passed to [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit).

# Arguments

  - `drtgt`: A [`PPCA`](@ref) dimension reduction target, specifying keyword arguments for PPCA.
  - `X`: Data matrix (observations × features) to which PPCA will be fitted.

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

Estimator for dimension reduction regression-based moment estimation.

`DimensionReductionRegression` is a flexible estimator type for performing regression with dimension reduction, such as PCA or PPCA, as a preprocessing step. It allows users to specify the expected returns estimator, variance estimator, dimension reduction target (e.g., `PCA`, `PPCA`), and the regression target (e.g., `LinearModel`).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DimensionReductionRegression(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        drtgt::DimensionReductionTarget = PCA(),
        retgt::AbstractRegressionTarget = LinearModel()
    ) -> DimensionReductionRegression

Keywords correspond to the struct's fields.

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

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
@concrete struct DimensionReductionRegression <: AbstractRegressionEstimator
    "$(field_dict[:ve])"
    ve
    "$(field_dict[:drtgt])"
    drtgt
    "$(field_dict[:dretgt])"
    retgt
    function DimensionReductionRegression(ve::AbstractVarianceEstimator,
                                          drtgt::DimensionReductionTarget,
                                          retgt::AbstractRegressionTarget)
        if haskey(retgt.kwargs, :weights)
            @argcheck(isa(retgt.kwargs.weights, ObsWeights), TypeError)
            if isa(retgt.kwargs.weights, AbstractVector)
                @argcheck(!isempty(retgt.kwargs.weights), IsEmptyError)
            end
        end
        return new{typeof(ve), typeof(drtgt), typeof(retgt)}(ve, drtgt, retgt)
    end
end
function DimensionReductionRegression(; ve::AbstractVarianceEstimator = SimpleVariance(),
                                      drtgt::DimensionReductionTarget = PCA(),
                                      retgt::AbstractRegressionTarget = LinearModel())
    return DimensionReductionRegression(ve, drtgt, retgt)
end
function factory(re::DimensionReductionRegression, w::ObsWeights)
    return DimensionReductionRegression(; ve = factory(re.ve, w),
                                        drtgt = factory(re.drtgt, w),
                                        retgt = factory(re.retgt, w))
end
function regression_view(re::DimensionReductionRegression, i)
    return DimensionReductionRegression(; ve = moment_view(re.ve, i), drtgt = re.drtgt,
                                        retgt = re.retgt)
end
"""
    prep_dim_red_reg(drtgt::DimensionReductionTarget, X::MatNum)

Prepare data for dimension reduction regression.

This helper function standardizes the feature matrix `X` (using Z-score normalization), fits the specified dimension reduction model (e.g., PCA or PPCA), and projects the standardized data into the reduced-dimensional space. It returns the projected data (with an intercept column) and the projection matrix.

# Arguments

  - `drtgt`: Dimension reduction target (e.g., `PCA()`, `PPCA()`).
  - `X`: Feature matrix (observations × features) to be reduced.

# Returns

  - `x1::MatNum`: Projected feature matrix with an intercept column prepended.
  - `Vp::MatNum`: Projection matrix from the fitted dimension reduction model.

# Details

  - Standardizes `X` using Z-score normalization (mean 0, variance 1).
  - Fits the dimension reduction model specified by `drtgt` to the standardized data.
  - Projects the standardized data into the reduced space.
  - Prepends a column of ones to the projected data for use as an intercept in regression.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
"""
function prep_dim_red_reg(drtgt::DimensionReductionTarget, X::MatNum)
    N = size(X, 1)
    X_std = StatsBase.standardize(StatsBase.ZScoreTransform, transpose(X); dims = 2)
    model = StatsAPI.fit(drtgt, X_std)
    Xp = transpose(StatsAPI.predict(model, X_std))
    Vp = MultivariateStats.projection(model)
    x1 = [ones(eltype(X), N) Xp]
    return x1, Vp
end
"""
    _regression(re::DimensionReductionRegression, y::VecNum, mu::VecNum,
               sigma::VecNum, x1::MatNum, Vp::MatNum)

Fit a regression model in reduced-dimensional space and recover coefficients in the original feature space.

This function fits a regression model (as specified by `retgt`) to the response vector `y` using the projected feature matrix `x1` (typically obtained from a dimension reduction method such as PCA or PPCA). It then transforms the estimated coefficients from the reduced space back to the original feature space using the projection matrix `Vp` and rescales them by the standard deviations `sigma`. The intercept is adjusted to account for the mean of `y` and the means of the original features.

# Arguments

  - `re`: Dimension reduction regression.
  - `y`: Response vector.
  - `mu`: Mean vector of the original features.
  - `sigma`: Standard deviation vector of the original features.
  - `x1`: Projected feature matrix with intercept column (from dimension reduction).
  - `Vp`: Projection matrix from the fitted dimension reduction model.

# Returns

  - `beta::VecNum`: Vector of regression coefficients in the original feature space, with the intercept as the first element.

# Details

  - Fits the regression model in the reduced space using `x1` and `y`.
  - Extracts the coefficients for the principal components (excluding the intercept).
  - Transforms the coefficients back to the original feature space using `Vp` and rescales by `sigma`.
  - Computes the intercept so that predictions are unbiased with respect to the means.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`prep_dim_red_reg`](@ref)
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

This method fits a regression model with dimension reduction (e.g., PCA or PPCA) to each column of the response matrix `X`, using the feature matrix `F` as predictors. For each response vector (column of `X`), the features are first standardized and projected into a lower-dimensional space using the dimension reduction target specified in `re.drtgt`. A regression model (specified by `re.retgt`) is then fitted in the reduced space, and the coefficients are mapped back to the original feature space.

# Arguments

  - `re`: Dimension reduction regression estimator specifying the expected returns estimator, variance estimator, dimension reduction target, and regression target.
  - `X`: Response matrix (observations × targets/assets).
  - `F`: Feature matrix (observations × features).

# Returns

  - `reg::Regression`: A regression result object containing:

      + `b`: Vector of intercepts for each response.
      + `M`: Matrix of coefficients for each response and feature (in the original feature space).
      + `L`: Matrix of coefficients in the reduced (projected) space.

# Details

  - For each column in `X`, the features in `F` are standardized, projected using the dimension reduction model, and a regression is fitted in the reduced space.
  - The resulting coefficients are transformed back to the original feature space and rescaled.
  - The output `Regression` object contains the intercepts, coefficient matrix in the original space, and the projected coefficients.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`prep_dim_red_reg`](@ref)
  - [`Regression`](@ref)
"""
function regression(re::DimensionReductionRegression, X::MatNum, F::MatNum)
    cols = size(F, 2) + 1
    rows = size(X, 2)
    rr = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    f1, Vp = prep_dim_red_reg(re.drtgt, F)
    me = ifelse(isnothing(re.ve.me), SimpleExpectedReturns(), re.ve.me)
    mu = Statistics.mean(me, F; dims = 1)
    sigma = vec(Statistics.std(re.ve, F; dims = 1))
    mu = vec(mu)
    for i in axes(rr, 1)
        rr[i, :] = _regression(re, view(X, :, i), mu, sigma, f1, Vp)
    end
    b = view(rr, :, 1)
    M = view(rr, :, 2:cols)
    L = transpose(LinearAlgebra.pinv(Vp) * transpose(M .* transpose(sigma)))
    return Regression(; b = b, M = M, L = L)
end

export PCA, PPCA, DimensionReductionRegression
