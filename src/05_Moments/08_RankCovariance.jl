"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all rank-based covariance estimators.

A subtype computes its correlation matrix from the ranks of the returns. [`KendallCovariance`](@ref) and [`SpearmanCovariance`](@ref) are its two subtypes. The method of [`show_fields`](@ref) on this type renders the field `ve` of each subtype and leaves out its `cache`.

# Related

  - [`KendallCovariance`](@ref)
  - [`SpearmanCovariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`show_fields`](@ref)
"""
abstract type RankCovarianceEstimator <: AbstractCovarianceEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Measures monotonic association with Kendall's tau, which counts concordant pairs against discordant pairs.

Kendall's tau reads only the ranks of the returns. An outlier changes it only through its rank, and the statistic does not assume Gaussian data. The generic covariance fallback rescales the correlation matrix by the marginal standard deviations of `ve`. Entry ``(i, j)`` of the covariance is ``\\hat{\\tau}^b_{ij}\\,\\hat{\\sigma}_i\\,\\hat{\\sigma}_j``, and the diagonal is the variance.

Equation 1 of [kendall1938](@cite) states the coefficient as the difference of the counts of concordant and discordant pairs, divided by the number of pairs. Section 6.1.3 of [cajas2025](@cite) states it again. The library computes the tie-corrected form of Equations 10 and 12 of [kendall1945](@cite). The two forms differ when the data have ties.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KendallCovariance(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> KendallCovariance

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ve`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ve`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:kendall1938]) Section 2, Equation 1.
  - $(ref_dict[:kendall1945]) Sections 10 to 11, Equations 10 and 12.
  - $(ref_dict[:cajas2025]) Section 6.1.3, Equation 6.3.
"""
@propagatable @concrete struct KendallCovariance <: RankCovarianceEstimator
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function KendallCovariance(ve::AbstractVarianceEstimator,
                               cache::Option{<:AbstractPartialFitState})
        return new{typeof(ve), typeof(cache)}(ve, cache)
    end
end
function KendallCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                           cache::Option{<:AbstractPartialFitState} = nothing)::KendallCovariance
    return KendallCovariance(ve, cache)
end
"""
    Statistics.cor(::KendallCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Kendall's tau rank correlation matrix with a [`KendallCovariance`](@ref) estimator.

Entry ``(i, j)`` is the Kendall's tau of the returns of assets ``i`` and ``j``, which measures their monotonic association.

# Mathematical definition

For two asset return series ``(x_1, \\ldots, x_T)`` and ``(y_1, \\ldots, y_T)``, `StatsBase.corkendall` computes the tie-corrected ``\\tau_b``:

```math
\\begin{align}
\\hat{\\tau}^b_{ij} &= \\frac{C - D}{\\sqrt{(n_0 - n_x)(n_0 - n_y)}}\\,, \\\\
n_0 &= \\binom{T}{2}\\,, \\quad n_x = \\sum_{g} \\binom{t_g}{2}\\,, \\quad n_y = \\sum_{h} \\binom{u_h}{2}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\tau}^b_{ij}``: The Kendall's ``\\tau_b`` rank correlation of assets ``i`` and ``j``.
  - ``C``: The number of concordant pairs. A pair ``(t, s)`` is concordant if ``(x_t - x_s)(y_t - y_s) > 0``.
  - ``D``: The number of discordant pairs. A pair ``(t, s)`` is discordant if ``(x_t - x_s)(y_t - y_s) < 0``.
  - ``n_0``: The total number of pairs.
  - ``t_g``, ``u_h``: The sizes of the ``g``-th group of tied ``x`` values and of the ``h``-th group of tied ``y`` values.
  - $(math_dict[:T])

Without ties, ``n_x = n_y = 0`` and ``\\hat{\\tau}^b`` reduces to ``\\tau_a = (C - D) / \\binom{T}{2}``, which is Equation 1 of [kendall1938](@cite). With ties, the two differ, because the tie counts make the denominator smaller. Equation 10 of [kendall1945](@cite) gives a tied pair the score zero, and Equation 12 then gives the denominator of the formula above. On the ten ranks of Example 3 of that work, the library returns ``33 / \\sqrt{41 \\cdot 36} \\approx 0.85896``. The work truncates this value to 0.8589.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), which transposes it when `dims` is `2` and refuses any other value.
 2. Refuse a gapped sample with [`assert_finite_sample`](@ref).
 3. Return `StatsBase.corkendall(X)`, the tie-corrected ``\\tau_b`` of each pair of columns. The method does not scale or clamp the result. The diagonal is exactly `1`, also for a constant column, and the other entries of a constant column are `NaN`.

# Arguments

  - `ce`: The Kendall's tau covariance estimator.
  - `X`: The data matrix of asset returns, `observations × assets`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method ignores.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `rho::Matrix{<:Number}`: The symmetric matrix of Kendall's tau rank correlation coefficients.

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
    assert_finite_sample(X)
    return StatsBase.corkendall(X)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures monotonic association with Spearman's rho, the Pearson correlation of the rank-transformed returns.

Spearman's rho reads only the ranks of the returns. An outlier changes it only through its rank, and the statistic does not assume Gaussian data. The generic covariance fallback rescales the correlation matrix by the marginal standard deviations of `ve`. Entry ``(i, j)`` of the covariance is ``\\hat{\\rho}^S_{ij}\\,\\hat{\\sigma}_i\\,\\hat{\\sigma}_j``, and the diagonal is the variance.

Part I, Section 3 of [spearman1904](@cite) computes the correlation of the ranks by the method of product moments. It gives the reason for the ranks. A rank does not depend on the form of the marginal distributions, and an extreme value changes it little. Section 4(d) of that part gives a different statistic, ``R = 1 - 3 S_d / (n^{2} - 1)``, from the sum of absolute rank differences ``S_d``. This estimator does not compute that statistic.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SpearmanCovariance(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> SpearmanCovariance

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ve`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ve`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:spearman1904]) Part I, Sections 3 and 4(d), pages 79 to 87.
"""
@propagatable @concrete struct SpearmanCovariance <: RankCovarianceEstimator
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function SpearmanCovariance(ve::AbstractVarianceEstimator,
                                cache::Option{<:AbstractPartialFitState})
        return new{typeof(ve), typeof(cache)}(ve, cache)
    end
end
function SpearmanCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                            cache::Option{<:AbstractPartialFitState} = nothing)::SpearmanCovariance
    return SpearmanCovariance(ve, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`RankCovarianceEstimator`](@ref), such as a [`KendallCovariance`](@ref) or a [`SpearmanCovariance`](@ref), except `cache`.

A `cache` holds the state of an incremental fit. That state is not part of the configuration, and it prints under the estimator at each site that renders the estimator. Set `set_show_nothing_fields!(:KendallCovariance, true)` or `set_show_nothing_fields!(:SpearmanCovariance, true)` to render it for that type.

# Arguments

  - `::RankCovarianceEstimator`: The rank covariance estimator. The method reads only its type.

# Returns

  - `fields::Tuple`: The names of the fields to render, `(:ve,)`.

# Related

  - [`KendallCovariance`](@ref)
  - [`SpearmanCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::RankCovarianceEstimator) = (:ve,)
"""
    Statistics.cor(::SpearmanCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the Spearman's rho rank correlation matrix with a [`SpearmanCovariance`](@ref) estimator.

Entry ``(i, j)`` is the Spearman's rho of the returns of assets ``i`` and ``j``, which measures their monotonic association.

# Mathematical definition

Spearman's ``\\rho`` is the Pearson correlation of the ranks of the data. Let ``\\mathrm{rk}(x_t)`` be the mid-rank of observation ``x_t`` among ``x_1, \\ldots, x_T``. Each value of a group of tied values gets the average rank of the group:

```math
\\begin{align}
\\hat{\\rho}^S_{ij} &= \\frac{\\mathrm{cov}\\!\\left(\\mathrm{rk}(x_{\\cdot i}),\\, \\mathrm{rk}(x_{\\cdot j})\\right)}{\\sigma_{\\mathrm{rk}(x_{\\cdot i})} \\, \\sigma_{\\mathrm{rk}(x_{\\cdot j})}}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\rho}^S_{ij}``: The Spearman's ``\\rho`` rank correlation of assets ``i`` and ``j``.
  - $(math_dict[:T])
  - ``x_{ti}``: The return of asset ``i`` at time ``t``.
  - ``\\mathrm{rk}(\\cdot)``: The mid-rank function.
  - ``\\sigma_{\\mathrm{rk}(\\cdot)}``: The standard deviation of the ranks.

Without ties, this is equal to the closed form ``1 - 6 \\sum_t d_t^2 / (T(T^2 - 1))``, with ``d_t = \\mathrm{rk}(x_{ti}) - \\mathrm{rk}(x_{tj})``. With ties, the two differ.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), which transposes it when `dims` is `2` and refuses any other value.
 2. Refuse a gapped sample with [`assert_finite_sample`](@ref).
 3. Return `StatsBase.corspearman(X)`, the Pearson correlation of the mid-ranked columns. The method does not scale or clamp the result. The diagonal is exactly `1`, also for a constant column, and the other entries of a constant column are `NaN`.

# Arguments

  - `ce`: The Spearman's rho covariance estimator.
  - `X`: The data matrix of asset returns, `observations × assets`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method ignores.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `rho::Matrix{<:Number}`: The symmetric matrix of Spearman's rho rank correlation coefficients.

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
    assert_finite_sample(X)
    return StatsBase.corspearman(X)
end
export KendallCovariance, SpearmanCovariance
