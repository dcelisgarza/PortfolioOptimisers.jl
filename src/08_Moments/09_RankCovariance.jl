"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all rank-based covariance estimators.

All concrete and/or abstract types implementing rank-based covariance estimation algorithms should be subtypes of `RankCovarianceEstimator`.

# Related

  - [`KendallCovariance`](@ref)
  - [`SpearmanCovariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type RankCovarianceEstimator <: AbstractCovarianceEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Robust covariance estimator based on Kendall's tau rank correlation.

`KendallCovariance` implements a covariance estimator that uses Kendall's tau rank correlation to measure the monotonic association between pairs of asset returns. This estimator is robust to outliers and non-Gaussian data, making it suitable for financial applications where heavy tails or non-linear dependencies are present.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KendallCovariance(;
        ve::AbstractVarianceEstimator = SimpleVariance()
    ) -> KendallCovariance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> KendallCovariance()
KendallCovariance
  ve ┼ SimpleVariance
     │          me ┼ SimpleExpectedReturns
     │             │   w ┴ nothing
     │           w ┼ nothing
     │   corrected ┴ Bool: true
```

# Related

  - [`RankCovarianceEstimator`](@ref)
  - [`SpearmanCovariance`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`SimpleVariance`](@ref)
"""
@propagatable @concrete struct KendallCovariance <: RankCovarianceEstimator
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    function KendallCovariance(ve::AbstractVarianceEstimator)
        return new{typeof(ve)}(ve)
    end
end
function KendallCovariance(;
                           ve::AbstractVarianceEstimator = SimpleVariance())::KendallCovariance
    return KendallCovariance(ve)
end
"""
    Statistics.cor(::KendallCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Kendall's tau rank correlation matrix using a [`KendallCovariance`](@ref) estimator.

This method computes the pairwise Kendall's tau rank correlation matrix for the input data matrix `X`. Kendall's tau measures the monotonic association between pairs of asset returns and is robust to outliers and non-Gaussian data.

# Mathematical definition

For two asset return series ``(x_1, \\ldots, x_T)`` and ``(y_1, \\ldots, y_T)``, Kendall's ``\\tau`` is:

```math
\\begin{align}
\\hat{\\tau}_{ij} &= \\frac{C - D}{\\binom{T}{2}}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\tau}_{ij}``: Kendall's ``\\tau`` rank correlation between assets ``i`` and ``j``.
  - ``C``: Number of concordant pairs; a pair ``(t, s)`` is concordant if ``(x_t - x_s)(y_t - y_s) > 0``.
  - ``D``: Number of discordant pairs; a pair ``(t, s)`` is discordant if ``(x_t - x_s)(y_t - y_s) < 0``.
  - $(math_dict[:T])

# Arguments

  - `ce`: Kendall's tau-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of Kendall's tau rank correlation coefficients.

# Validation

  - `dims` is either `1` or `2`.

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cor(KendallCovariance(), X)
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```

# Related

  - [`KendallCovariance`](@ref)
  - [`corkendall`](https://juliastats.org/StatsBase.jl/stable/ranking/#StatsBase.corkendall)
"""
function Statistics.cor(::KendallCovariance, X::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
    end
    return StatsBase.corkendall(X)
end
"""
$(DocStringExtensions.TYPEDEF)

Robust covariance estimator based on Spearman's rho rank correlation.

`SpearmanCovariance` implements a covariance estimator that uses Spearman's rho rank correlation to measure the monotonic association between pairs of asset returns. This estimator is robust to outliers and non-Gaussian data, making it suitable for financial applications where heavy tails or non-linear dependencies are present.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SpearmanCovariance(;
        ve::AbstractVarianceEstimator = SimpleVariance()
    ) -> SpearmanCovariance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> SpearmanCovariance()
SpearmanCovariance
  ve ┼ SimpleVariance
     │          me ┼ SimpleExpectedReturns
     │             │   w ┴ nothing
     │           w ┼ nothing
     │   corrected ┴ Bool: true
```

# Related

  - [`RankCovarianceEstimator`](@ref)
  - [`KendallCovariance`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`SimpleVariance`](@ref)
"""
@propagatable @concrete struct SpearmanCovariance <: RankCovarianceEstimator
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    function SpearmanCovariance(ve::AbstractVarianceEstimator)
        return new{typeof(ve)}(ve)
    end
end
function SpearmanCovariance(;
                            ve::AbstractVarianceEstimator = SimpleVariance())::SpearmanCovariance
    return SpearmanCovariance(ve)
end
"""
    Statistics.cor(::SpearmanCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Spearman's rho rank correlation matrix using a [`SpearmanCovariance`](@ref) estimator.

This method computes the pairwise Spearman's rho rank correlation matrix for the input data matrix `X`. Spearman's rho measures the monotonic association between pairs of asset returns and is robust to outliers and non-Gaussian data.

# Mathematical definition

Spearman's ``\\rho`` is the Pearson correlation of the rank-transformed data. Let ``\\mathrm{rk}(x_t)`` denote the rank of observation ``x_t`` among ``x_1, \\ldots, x_T``:

```math
\\begin{align}
\\hat{\\rho}^S_{ij} &= 1 - \\frac{6 \\sum_{t=1}^{T} d_t^2}{T(T^2 - 1)}, \\quad d_t = \\mathrm{rk}(x_{ti}) - \\mathrm{rk}(x_{tj})\\,.
\\end{align}
```

Where:

  - ``\\hat{\\rho}^S_{ij}``: Spearman's ``\\rho`` rank correlation between assets ``i`` and ``j``.
  - $(math_dict[:T])
  - ``x_{ti}``: Return of asset ``i`` at time ``t``.
  - ``d_t``: Difference in ranks between assets ``i`` and ``j`` at time ``t``.
  - ``\\mathrm{rk}(\\cdot)``: Rank function.

# Arguments

  - `ce`: Spearman's rho-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of Spearman's rho rank correlation coefficients.

# Validation

  - `dims` is either `1` or `2`.

# Examples

```jldoctest
julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cor(SpearmanCovariance(), X)
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```

# Related

  - [`SpearmanCovariance`](@ref)
  - [`corspearman`](https://juliastats.org/StatsBase.jl/stable/ranking/#StatsBase.corspearman)
"""
function Statistics.cor(::SpearmanCovariance, X::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
    end
    return StatsBase.corspearman(X)
end
export KendallCovariance, SpearmanCovariance
