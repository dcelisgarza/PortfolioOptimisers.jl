"""
    abstract type DimensionReductionTarget <: AbstractRegressionAlgorithm end

Abstract supertype for all dimension reduction regression algorithm targets in PortfolioOptimisers.jl.

All concrete types implementing dimension reduction algorithms for regression (such as PCA or PPCA) should subtype `DimensionReductionTarget`. This enables a consistent and extensible interface for specifying dimension reduction strategies within regression-based moment estimation.

These types are used to specify the dimension reduction method when constructing a [`DimensionReductionRegression`](@ref) estimator.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
  - [`PPCA`](@ref)
  - [`AbstractRegressionAlgorithm`](@ref)
"""
abstract type DimensionReductionTarget <: AbstractRegressionAlgorithm end

"""
    struct PCA{T1} <: DimensionReductionTarget
        kwargs::T1
    end

Principal Component Analysis (PCA) dimension reduction target.

`PCA` is used to specify principal component analysis as the dimension reduction method for regression-based moment estimation. The `kwargs` field stores keyword arguments to be passed to the underlying PCA implementation (e.g., from `MultivariateStats.jl`).

# Fields

  - `kwargs`: Keyword arguments for [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit).

# Related

  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`PPCA`](@ref)
"""
struct PCA{T1} <: DimensionReductionTarget
    kwargs::T1
end
"""
    PCA(; kwargs::NamedTuple = (;))

Construct a [`PCA`](@ref) dimension reduction target.

This constructor creates a `PCA` object with the specified keyword arguments, which are passed to [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit).

# Arguments

  - `kwargs`: Keyword arguments for configuring the PCA algorithm.

# Returns

  - `PCA`: A PCA dimension reduction target.

# Constructor

    PCA(; kwargs::NamedTuple = (;))

# Examples

```jldoctest
julia> PCA()
PCA
  kwargs | @NamedTuple{}: NamedTuple()
```

# Related

  - [`PCA`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
function PCA(; kwargs::NamedTuple = (;))
    return PCA(kwargs)
end

"""
    StatsAPI.fit(drtgt::PCA, X::AbstractMatrix)

Fit a Principal Component Analysis (PCA) model to the data matrix `X` using the configuration in `drtgt`.

This method applies PCA as a dimension reduction technique for regression-based moment estimation. The keyword arguments stored in `drtgt.kwargs` are passed to [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit).

# Arguments

  - `drtgt`: A [`PCA`](@ref) dimension reduction target, specifying keyword arguments for PCA.
  - `X`: Data matrix (observations × features) to which PCA will be fitted.

# Returns

  - `model`: A fitted PCA model object from `MultivariateStats.jl`.

# Related

  - [`PCA`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
function StatsAPI.fit(drtgt::PCA, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PCA, X; drtgt.kwargs...)
end

"""
    struct PPCA{T1} <: DimensionReductionTarget
        kwargs::T1
    end

Probabilistic Principal Component Analysis (PPCA) dimension reduction target.

`PPCA` is used to specify probabilistic principal component analysis as the dimension reduction method for regression-based moment estimation. The `kwargs` field stores keyword arguments to be passed to the underlying PPCA implementation (e.g., from `MultivariateStats.jl`).

# Fields

  - `kwargs`: Keyword arguments for [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit).

# Constructor

    PPCA(; kwargs::NamedTuple = ())

# Related

  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
  - [`PCA`](@ref)
"""
struct PPCA{T1} <: DimensionReductionTarget
    kwargs::T1
end
"""
    PPCA(; kwargs::NamedTuple = (;))

Construct a [`PPCA`](@ref) dimension reduction target.

This constructor creates a `PPCA` object with the specified keyword arguments, which are passed to the underlying PPCA algorithm.

# Arguments

  - `kwargs`: Keyword arguments for [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit).

# Returns

  - `PPCA`: A PPCA dimension reduction target.

# Examples

```jldoctest
julia> PPCA()
PPCA
  kwargs | @NamedTuple{}: NamedTuple()
```

# Related

  - [`PPCA`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
function PPCA(; kwargs::NamedTuple = (;))
    return PPCA(kwargs)
end

"""
    StatsAPI.fit(drtgt::PPCA, X::AbstractMatrix)

Fit a Probabilistic Principal Component Analysis (PPCA) model to the data matrix `X` using the configuration in `drtgt`.

This method applies PPCA as a dimension reduction technique for regression-based moment estimation. The keyword arguments stored in `drtgt.kwargs` are passed to [`MultivariateStats.fit`](https://juliastats.org/MultivariateStats.jl/stable/pca/#StatsAPI.fit).

# Arguments

  - `drtgt`: A [`PPCA`](@ref) dimension reduction target, specifying keyword arguments for PPCA.
  - `X`: Data matrix (observations × features) to which PPCA will be fitted.

# Returns

  - `model`: A fitted PPCA model object from `MultivariateStats.jl`.

# Related

  - [`PPCA`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`DimensionReductionRegression`](@ref)
"""
function StatsAPI.fit(drtgt::PPCA, X::AbstractMatrix)
    return MultivariateStats.fit(MultivariateStats.PPCA, X; drtgt.kwargs...)
end

"""
    struct DimensionReductionRegression{T1, T2, T3, T4} <: AbstractRegressionEstimator
        me::T1
        ve::T2
        drtgt::T3
        retgt::T4
    end

Estimator for dimension reduction regression-based moment estimation.

`DimensionReductionRegression` is a flexible estimator type for performing regression with dimension reduction, such as PCA or PPCA, as a preprocessing step. It allows users to specify the expected returns estimator, variance estimator, dimension reduction target (e.g., `PCA`, `PPCA`), and the regression target (e.g., `LinearModel`). This enables modular workflows for moment estimation in high-dimensional settings.

# Fields

  - `me`: Expected returns estimator.
  - `ve`: Variance estimator.
  - `drtgt`: Dimension reduction target.
  - `retgt`: Regression target type.

# Constructor

    DimensionReductionRegression(;
                                 me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                 ve::AbstractVarianceEstimator = SimpleVariance(),
                                 drtgt::DimensionReductionTarget = PCA(),
                                 retgt::AbstractRegressionTarget = LinearModel())

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
struct DimensionReductionRegression{T1,T2,T3,T4} <: AbstractRegressionEstimator
    me::T1
    ve::T2
    drtgt::T3
    retgt::T4
end
"""
    DimensionReductionRegression(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                                  ve::AbstractVarianceEstimator = SimpleVariance(),
                                  drtgt::DimensionReductionTarget = PCA(),
                                  retgt::AbstractRegressionTarget = LinearModel())

Construct a [`DimensionReductionRegression`](@ref) estimator for regression-based moment estimation with dimension reduction.

This constructor creates a `DimensionReductionRegression` object with the specified expected returns estimator, variance estimator, dimension reduction target, and regression target.

# Arguments

  - `me`: Expected returns estimator.
  - `ve`: Variance estimator.
  - `drtgt`: Dimension reduction target.
  - `retgt`: Regression target type.

# Returns

  - `DimensionReductionRegression`: A configured dimension reduction regression estimator.

# Examples

```jldoctest
julia> DimensionReductionRegression()
DimensionReductionRegression
     me | SimpleExpectedReturns
        |   w | nothing
     ve | SimpleVariance
        |          me | SimpleExpectedReturns
        |             |   w | nothing
        |           w | nothing
        |   corrected | Bool: true
  drtgt | PCA
        |   kwargs | @NamedTuple{}: NamedTuple()
  retgt | LinearModel
        |   kwargs | @NamedTuple{}: NamedTuple()
```

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`DimensionReductionTarget`](@ref)
  - [`AbstractRegressionTarget`](@ref)
"""
function DimensionReductionRegression(;
    me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
    ve::AbstractVarianceEstimator = SimpleVariance(),
    drtgt::DimensionReductionTarget = PCA(),
    retgt::AbstractRegressionTarget = LinearModel(),
)
    return DimensionReductionRegression(me, ve, drtgt, retgt)
end

"""
    prep_dim_red_reg(drtgt::DimensionReductionTarget, X::AbstractMatrix)

Prepare data for dimension reduction regression.

This helper function standardizes the feature matrix `X` (using Z-score normalization), fits the specified dimension reduction model (e.g., PCA or PPCA), and projects the standardized data into the reduced-dimensional space. It returns the projected data (with an intercept column) and the projection matrix.

# Arguments

  - `drtgt`: Dimension reduction target (e.g., `PCA()`, `PPCA()`).
  - `X`: Feature matrix (observations × features) to be reduced.

# Returns

  - `x1::AbstractMatrix{<:Real}`: Projected feature matrix with an intercept column prepended.
  - `Vp::AbstractMatrix{<:Real}`: Projection matrix from the fitted dimension reduction model.

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
function prep_dim_red_reg(drtgt::DimensionReductionTarget, X::AbstractMatrix)
    N = size(X, 1)
    X_std = StatsBase.standardize(StatsBase.ZScoreTransform, transpose(X); dims = 2)
    model = fit(drtgt, X_std)
    Xp = transpose(predict(model, X_std))
    Vp = projection(model)
    x1 = [ones(eltype(X), N) Xp]
    return x1, Vp
end

"""
    regression(retgt::AbstractRegressionTarget, y::AbstractVector, mu::AbstractVector,
               sigma::AbstractVector, x1::AbstractMatrix, Vp::AbstractMatrix)

Fit a regression model in reduced-dimensional space and recover coefficients in the original feature space.

This function fits a regression model (as specified by `retgt`) to the response vector `y` using the projected feature matrix `x1` (typically obtained from a dimension reduction method such as PCA or PPCA). It then transforms the estimated coefficients from the reduced space back to the original feature space using the projection matrix `Vp` and rescales them by the standard deviations `sigma`. The intercept is adjusted to account for the mean of `y` and the means of the original features.

# Arguments

  - `retgt`: Regression target type (e.g., `LinearModel()`).
  - `y`: Response vector.
  - `mu`: Mean vector of the original features.
  - `sigma`: Standard deviation vector of the original features.
  - `x1`: Projected feature matrix with intercept column (from dimension reduction).
  - `Vp`: Projection matrix from the fitted dimension reduction model.

# Returns

  - `beta::Vector{<:Real}`: Vector of regression coefficients in the original feature space, with the intercept as the first element.

# Details

  - Fits the regression model in the reduced space using `x1` and `y`.
  - Extracts the coefficients for the principal components (excluding the intercept).
  - Transforms the coefficients back to the original feature space using `Vp` and rescales by `sigma`.
  - Computes the intercept so that predictions are unbiased with respect to the means.

# Related

  - [`DimensionReductionRegression`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`prep_dim_red_reg`](@ref)
"""
function regression(
    retgt::AbstractRegressionTarget,
    y::AbstractVector,
    mu::AbstractVector,
    sigma::AbstractVector,
    x1::AbstractMatrix,
    Vp::AbstractMatrix,
)
    fit_result = fit(retgt, x1, y)
    beta_pc = coef(fit_result)[2:end]
    beta = Vp * beta_pc ./ sigma
    beta0 = mean(y) - dot(beta, mu)
    pushfirst!(beta, beta0)
    return beta
end

"""
    regression(re::DimensionReductionRegression, X::AbstractMatrix, F::AbstractMatrix)

Apply dimension reduction regression to each column of a response matrix.

This method fits a regression model with dimension reduction (e.g., PCA or PPCA) to each column of the response matrix `X`, using the feature matrix `F` as predictors. For each response vector (column of `X`), the features are first standardized and projected into a lower-dimensional space using the dimension reduction target specified in `re.drtgt`. A regression model (specified by `re.retgt`) is then fitted in the reduced space, and the coefficients are mapped back to the original feature space.

# Arguments

  - `re`: Dimension reduction regression estimator specifying the expected returns estimator, variance estimator, dimension reduction target, and regression target.
  - `X`: Response matrix (observations × targets/assets).
  - `F`: Feature matrix (observations × features).

# Returns

  - `Regression`: A regression result object containing:

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
function regression(re::DimensionReductionRegression, X::AbstractMatrix, F::AbstractMatrix)
    cols = size(F, 2) + 1
    rows = size(X, 2)
    rr = zeros(promote_type(eltype(F), eltype(X)), rows, cols)
    f1, Vp = prep_dim_red_reg(re.drtgt, F)
    mu = mean(re.me, F; dims = 1)
    sigma = vec(std(re.ve, F; mean = mu, dims = 1))
    mu = vec(mu)
    for i in axes(rr, 1)
        rr[i, :] .= regression(re.retgt, view(X, :, i), mu, sigma, f1, Vp)
    end
    b = view(rr, :, 1)
    M = view(rr, :, 2:cols)
    L = transpose(pinv(Vp) * transpose(M .* transpose(sigma)))
    return Regression(; b = b, M = M, L = L)
end

export PCA, PPCA, DimensionReductionRegression
