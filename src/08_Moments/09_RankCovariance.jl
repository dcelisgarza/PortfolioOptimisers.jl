"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all rank-based covariance estimators in `PortfolioOptimisers.jl`.

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
@concrete struct KendallCovariance <: RankCovarianceEstimator
    "$(field_dict[:ve])"
    ve
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
\\hat{\\tau}_{ij} = \\frac{C - D}{\\binom{T}{2}}
```

where ``C`` is the number of concordant pairs and ``D`` is the number of discordant pairs among all ``\\binom{T}{2}`` pairs of observations. A pair ``(t, s)`` is concordant if ``(x_t - x_s)(y_t - y_s) > 0`` and discordant if ``(x_t - x_s)(y_t - y_s) < 0``.

# Arguments

  - `ce`: Kendall's tau-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of Kendall's tau rank correlation coefficients.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`KendallCovariance`](@ref)
  - [`corkendall`](https://juliastats.org/StatsBase.jl/stable/ranking/#StatsBase.corkendall)
"""
function Statistics.cor(::KendallCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return StatsBase.corkendall(X)
end
"""
    Statistics.cov(ce::KendallCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Kendall's tau rank covariance matrix using a [`KendallCovariance`](@ref) estimator.

This method computes the covariance matrix for the input data matrix `X` by combining the Kendall's tau rank correlation matrix with the marginal standard deviations estimated by the variance estimator in `ce`. This approach is robust to outliers and non-Gaussian data.

# Arguments

  - `ce`: Kendall's tau-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the variance estimator.

# Returns

  - `sigma::Matrix{<:Number}`: Symmetric matrix of Kendall's tau rank covariances.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`KendallCovariance`](@ref)
  - [`corkendall`](https://juliastats.org/StatsBase.jl/stable/ranking/#StatsBase.corkendall)
"""
function Statistics.cov(ce::KendallCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sigma = StatsBase.corkendall(X)
    return StatsBase.cor2cov!(sigma, sd)
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
@concrete struct SpearmanCovariance <: RankCovarianceEstimator
    "$(field_dict[:ve])"
    ve
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
\\hat{\\rho}^S_{ij} = 1 - \\frac{6 \\sum_{t=1}^{T} d_t^2}{T(T^2 - 1)}, \\quad d_t = \\mathrm{rk}(x_{ti}) - \\mathrm{rk}(x_{tj})
```

Where ``T`` is the number of observations, ``x_{ti}`` is the return of asset ``i`` at time ``t``, and ``d_t`` is the difference of ranks at time ``t``.

# Arguments

  - `ce`: Spearman's rho-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of Spearman's rho rank correlation coefficients.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`SpearmanCovariance`](@ref)
  - [`corspearman`](https://juliastats.org/StatsBase.jl/stable/ranking/#StatsBase.corspearman)
"""
function Statistics.cor(::SpearmanCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return StatsBase.corspearman(X)
end
"""
    Statistics.cov(ce::SpearmanCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Spearman's rho rank covariance matrix using a [`SpearmanCovariance`](@ref) estimator.

This method computes the covariance matrix for the input data matrix `X` by combining the Spearman's rho rank correlation matrix with the marginal standard deviations estimated by the variance estimator in `ce`. This approach is robust to outliers and non-Gaussian data.

# Arguments

  - `ce`: Spearman's rho-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the variance estimator.

# Returns

  - `sigma::Matrix{<:Number}`: Symmetric matrix of Spearman's rho rank covariances.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`SpearmanCovariance`](@ref)
  - [`corspearman`](https://juliastats.org/StatsBase.jl/stable/ranking/#StatsBase.corspearman)
"""
function Statistics.cov(ce::SpearmanCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sigma = StatsBase.corspearman(X)
    return StatsBase.cor2cov!(sigma, sd)
end
for ce in traverse_concrete_subtypes(RankCovarianceEstimator)
    eval(quote
             function factory(ce::$(ce), w::ObsWeights)
                 return $(ce)(; ve = factory(ce.ve, w))
             end
             function moment_view(ce::$(ce), i)
                 return $(ce)(; ve = moment_view(ce.ve, i))
             end
         end)
end

export KendallCovariance, SpearmanCovariance
