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

Measures monotonic association with Kendall's tau, counting concordant against discordant pairs.

The rank statistic is robust to outliers and to non-Gaussian data. The covariance follows from the generic fallback, which rescales the correlation matrix by the marginal standard deviations of `ve`.

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

# References

  - $(ref_dict[:cajas2025]) Section 6.1.3, equation 6.3.
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

For two asset return series ``(x_1, \\ldots, x_T)`` and ``(y_1, \\ldots, y_T)``, `StatsBase.corkendall` computes the tie-corrected ``\\tau_b``:

```math
\\begin{align}
\\hat{\\tau}^b_{ij} &= \\frac{C - D}{\\sqrt{(n_0 - n_x)(n_0 - n_y)}}\\,, \\\\
n_0 &= \\binom{T}{2}\\,, \\quad n_x = \\sum_{g} \\binom{t_g}{2}\\,, \\quad n_y = \\sum_{h} \\binom{u_h}{2}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\tau}^b_{ij}``: Kendall's ``\\tau_b`` rank correlation between assets ``i`` and ``j``.
  - ``C``: Number of concordant pairs; a pair ``(t, s)`` is concordant if ``(x_t - x_s)(y_t - y_s) > 0``.
  - ``D``: Number of discordant pairs; a pair ``(t, s)`` is discordant if ``(x_t - x_s)(y_t - y_s) < 0``.
  - ``n_0``: Total number of pairs.
  - ``t_g``, ``u_h``: Sizes of the ``g``-th group of tied ``x`` values and the ``h``-th group of tied ``y`` values.
  - $(math_dict[:T])

Without ties, ``n_x = n_y = 0`` and ``\\hat{\\tau}^b`` reduces to ``\\tau_a = (C - D) / \\binom{T}{2}``, which is equation 6.3 of the source. **The two differ when ties are present**: on `[1.0 1.0; 2.0 1.0; 2.0 3.0; 4.0 4.0; 5.0 2.0]` (one tied pair in each series), ``\\tau_a`` is `0.4` and this method returns `0.4444444444444444`.

# Arguments

  - `ce`: Kendall's tau-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of Kendall's tau rank correlation coefficients.

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
    X = dims_oriented(dims, X)
    return StatsBase.corkendall(X)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures monotonic association with Spearman's rho, the Pearson correlation of the rank-transformed returns.

The rank transform is robust to outliers and to non-Gaussian data. The covariance follows from the generic fallback, which rescales the correlation matrix by the marginal standard deviations of `ve`.

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

# References

  - $(ref_dict[:cajas2025]) Section 6.1.2, equation 6.2.
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

Spearman's ``\\rho`` is the Pearson correlation of the rank-transformed data. Let ``\\mathrm{rk}(x_t)`` denote the mid-rank of observation ``x_t`` among ``x_1, \\ldots, x_T``, so that a group of tied values shares their average rank:

```math
\\begin{align}
\\hat{\\rho}^S_{ij} &= \\frac{\\mathrm{cov}\\!\\left(\\mathrm{rk}(x_{\\cdot i}),\\, \\mathrm{rk}(x_{\\cdot j})\\right)}{\\sigma_{\\mathrm{rk}(x_{\\cdot i})} \\, \\sigma_{\\mathrm{rk}(x_{\\cdot j})}}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\rho}^S_{ij}``: Spearman's ``\\rho`` rank correlation between assets ``i`` and ``j``.
  - $(math_dict[:T])
  - ``x_{ti}``: Return of asset ``i`` at time ``t``.
  - ``\\mathrm{rk}(\\cdot)``: Mid-rank function.
  - ``\\sigma_{\\mathrm{rk}(\\cdot)}``: Standard deviation of the rank variable.

Without ties this equals the closed form ``1 - 6 \\sum_t d_t^2 / (T(T^2 - 1))``, with ``d_t = \\mathrm{rk}(x_{ti}) - \\mathrm{rk}(x_{tj})``. **The two differ when ties are present**: on `[1.0 1.0; 2.0 1.0; 2.0 3.0; 4.0 4.0; 5.0 2.0]` (one tied pair in each series), the closed form gives `0.575` and this method returns `0.5526315789473685`.

# Arguments

  - `ce`: Spearman's rho-based covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of Spearman's rho rank correlation coefficients.

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
    X = dims_oriented(dims, X)
    return StatsBase.corspearman(X)
end
export KendallCovariance, SpearmanCovariance
