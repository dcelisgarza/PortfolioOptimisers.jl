"""
    abstract type RankCovarianceEstimator <: AbstractCovarianceEstimator end

Abstract supertype for all rank-based covariance estimators in PortfolioOptimisers.jl.

All concrete types implementing rank-based covariance estimation algorithms (such as Kendall's tau or Spearman's rho) should subtype `RankCovarianceEstimator`. This enables a consistent interface for rank-based covariance estimators throughout the package and allows for flexible extension and dispatch.

# Related

  - [`KendallCovariance`](@ref)
  - [`SpearmanCovariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type RankCovarianceEstimator <: AbstractCovarianceEstimator end

"""
    struct KendallCovariance{T1 <: AbstractVarianceEstimator} <: RankCovarianceEstimator

Robust covariance estimator based on Kendall's tau rank correlation.

`KendallCovariance` implements a covariance estimator that uses Kendall's tau rank correlation to measure the monotonic association between pairs of asset returns. This estimator is robust to outliers and non-Gaussian data, making it suitable for financial applications where heavy tails or non-linear dependencies are present.

# Fields

  - `ve::AbstractVarianceEstimator`: Variance estimator used to compute marginal standard deviations.

# Constructor

    KendallCovariance(; ve::AbstractVarianceEstimator = SimpleVariance())

Construct a `KendallCovariance` object with the specified variance estimator.

# Related

  - [`RankCovarianceEstimator`](@ref)
  - [`SpearmanCovariance`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`SimpleVariance`](@ref)
"""
struct KendallCovariance{T1 <: AbstractVarianceEstimator} <: RankCovarianceEstimator
    ve::T1
end
"""
    KendallCovariance(; ve::AbstractVarianceEstimator = SimpleVariance())

Construct a [`KendallCovariance`](@ref) estimator for robust rank-based covariance or correlation estimation using Kendall's tau.

This constructor creates a `KendallCovariance` object using the specified variance estimator. The estimator computes the covariance matrix by combining the Kendall's tau rank correlation matrix with the marginal standard deviations.

# Arguments

  - `ve::AbstractVarianceEstimator`: Variance estimator.

# Returns

  - `KendallCovariance`: A configured Kendall's tau-based covariance estimator.

# Examples

```jldoctest
julia> ce = KendallCovariance()
KendallCovariance
  ve | SimpleVariance
     |          me | SimpleExpectedReturns
     |             |   w | nothing
     |           w | nothing
     |   corrected | Bool: true
```

# Related

  - [`KendallCovariance`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`SimpleVariance`](@ref)
"""
function KendallCovariance(; ve::AbstractVarianceEstimator = SimpleVariance())
    return KendallCovariance{typeof(ve)}(ve)
end

"""
    cor(::KendallCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the Kendall's tau rank correlation matrix using a [`KendallCovariance`](@ref) estimator.

This method computes the pairwise Kendall's tau rank correlation matrix for the input data matrix `X`. Kendall's tau measures the monotonic association between pairs of asset returns and is robust to outliers and non-Gaussian data.

# Arguments

  - `ce::KendallCovariance`: Kendall's tau-based covariance estimator.
  - `X::AbstractMatrix`: Data matrix of asset returns (observations × assets).
  - `dims::Int`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{Float64}`: Symmetric matrix of Kendall's tau rank correlation coefficients.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Related

  - [`KendallCovariance`](@ref)
  - [`corkendall`](https://juliastats.org/StatsBase.jl/stable/ranking/#StatsBase.corkendall)
"""
function Statistics.cor(::KendallCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return corkendall(X)
end

"""
    cov(ce::KendallCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the Kendall's tau rank covariance matrix using a [`KendallCovariance`](@ref) estimator.

This method computes the covariance matrix for the input data matrix `X` by combining the Kendall's tau rank correlation matrix with the marginal standard deviations estimated by the variance estimator in `ce`. This approach is robust to outliers and non-Gaussian data.

# Arguments

  - `ce::KendallCovariance`: Kendall's tau-based covariance estimator.
  - `X::AbstractMatrix`: Data matrix of asset returns (observations × assets).
  - `dims::Int`: Dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the variance estimator.

# Returns

  - `sigma::Matrix{Float64}`: Symmetric matrix of Kendall's tau rank covariances.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Related

  - [`KendallCovariance`](@ref)
  - [`corkendall`](https://juliastats.org/StatsBase.jl/stable/ranking/#StatsBase.corkendall)
"""
function Statistics.cov(ce::KendallCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return corkendall(X) ⊙ (std_vec ⊗ std_vec)
end
function factory(ce::KendallCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return KendallCovariance(; ve = factory(ce.ve, w))
end

"""
    struct SpearmanCovariance{T1 <: AbstractVarianceEstimator} <: RankCovarianceEstimator

Robust covariance estimator based on Spearman's rho rank correlation.

`SpearmanCovariance` implements a covariance estimator that uses Spearman's rho rank correlation to measure the monotonic association between pairs of asset returns. This estimator is robust to outliers and non-Gaussian data, making it suitable for financial applications where heavy tails or non-linear dependencies are present.

# Fields

  - `ve::AbstractVarianceEstimator`: Variance estimator used to compute marginal standard deviations.

# Constructor

    SpearmanCovariance(; ve::AbstractVarianceEstimator = SimpleVariance())

Construct a `SpearmanCovariance` object with the specified variance estimator.

# Related

  - [`RankCovarianceEstimator`](@ref)
  - [`KendallCovariance`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`SimpleVariance`](@ref)
"""
struct SpearmanCovariance{T1 <: AbstractVarianceEstimator} <: RankCovarianceEstimator
    ve::T1
end
"""
    SpearmanCovariance(; ve::AbstractVarianceEstimator = SimpleVariance())

Construct a [`SpearmanCovariance`](@ref) estimator for robust rank-based covariance or correlation estimation using Spearman's rho.

This constructor creates a `SpearmanCovariance` object using the specified variance estimator. The estimator computes the covariance matrix by combining the Spearman's rho rank correlation matrix with the marginal standard deviations.

# Arguments

  - `ve::AbstractVarianceEstimator`: Variance estimator.

# Returns

  - `SpearmanCovariance`: A configured Spearman's rho-based covariance estimator.

# Examples

```jldoctest
julia> ce = SpearmanCovariance()
SpearmanCovariance
  ve | SimpleVariance
     |          me | SimpleExpectedReturns
     |             |   w | nothing
     |           w | nothing
     |   corrected | Bool: true
```

# Related

  - [`SpearmanCovariance`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`SimpleVariance`](@ref)
"""
function SpearmanCovariance(; ve::AbstractVarianceEstimator = SimpleVariance())
    return SpearmanCovariance{typeof(ve)}(ve)
end

"""
    cor(::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the Spearman's rho rank correlation matrix using a [`SpearmanCovariance`](@ref) estimator.

This method computes the pairwise Spearman's rho rank correlation matrix for the input data matrix `X`. Spearman's rho measures the monotonic association between pairs of asset returns and is robust to outliers and non-Gaussian data.

# Arguments

  - `ce::SpearmanCovariance`: Spearman's rho-based covariance estimator.
  - `X::AbstractMatrix`: Data matrix of asset returns (observations × assets).
  - `dims::Int`: Dimension along which to compute the correlation (1 = columns/assets, 2 = rows). Default is `1`.
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{Float64}`: Symmetric matrix of Spearman's rho rank correlation coefficients.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Related

  - [`SpearmanCovariance`](@ref)
  - [`corspearman`](https://juliastats.org/StatsBase.jl/stable/ranking/#StatsBase.corspearman)
"""
function Statistics.cor(::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return corspearman(X)
end

"""
    cov(ce::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the Spearman's rho rank covariance matrix using a [`SpearmanCovariance`](@ref) estimator.

This method computes the covariance matrix for the input data matrix `X` by combining the Spearman's rho rank correlation matrix with the marginal standard deviations estimated by the variance estimator in `ce`. This approach is robust to outliers and non-Gaussian data.

# Arguments

  - `ce::SpearmanCovariance`: Spearman's rho-based covariance estimator.
  - `X::AbstractMatrix`: Data matrix of asset returns (observations × assets).
  - `dims::Int`: Dimension along which to compute the covariance (1 = columns/assets, 2 = rows). Default is `1`.
  - `kwargs...`: Additional keyword arguments passed to the variance estimator.

# Returns

  - `sigma::Matrix{Float64}`: Symmetric matrix of Spearman's rho rank covariances.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Related

  - [`SpearmanCovariance`](@ref)
  - [`corspearman`](https://juliastats.org/StatsBase.jl/stable/ranking/#StatsBase.corspearman)
"""
function Statistics.cov(ce::SpearmanCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return corspearman(X) ⊙ (std_vec ⊗ std_vec)
end
function factory(ce::SpearmanCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return SpearmanCovariance(; ve = factory(ce.ve, w))
end

export KendallCovariance, SpearmanCovariance
