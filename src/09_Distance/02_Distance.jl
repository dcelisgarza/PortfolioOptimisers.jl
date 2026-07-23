"""
$(DocStringExtensions.TYPEDEF)

If power is not `nothing`, computes the generalised distance estimator.

# Mathematical definition

```math
\\begin{align}
_{g}d_{i,\\,j} &= s \\cdot \\left(d_{i,\\,j}\\right)^{p}\\\\
    s &= \\begin{cases}
        1/2 & \\text{if } p \\mod 2 \\neq 0\\\\
        1 & \\text{otherwise}
        \\end{cases}\\,,
\\end{align}
```

Where:

  - ``_{g}d_{i,\\,j}``: Generalised distance between assets ``i`` and ``j``.
  - ``d_{i,\\,j}``: Base distance computed using the specified distance algorithm.
  - ``p``: Integer power.
  - ``s``: Scaling factor (``s = 1/2`` if ``p \\bmod 2 \\neq 0``, else ``s = 1``).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Distance(;
        power::Option{<:Integer} = nothing,
        alg::AbstractDistanceAlgorithm = SimpleDistance()
    ) -> Distance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:dopower])

# Examples

```jldoctest
julia> Distance()
Distance
  power ┼ nothing
    alg ┴ SimpleDistance()
```

# Related

  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
"""
@concrete struct Distance <: AbstractDistanceEstimator
    """
    $(field_dict[:dopower])
    """
    power
    """
    $(field_dict[:dalg])
    """
    alg
    function Distance(power::Option{<:Integer}, alg::AbstractDistanceAlgorithm)
        if !isnothing(power)
            @argcheck(one(power) <= power, DomainError)
        end
        return new{typeof(power), typeof(alg)}(power, alg)
    end
end
function Distance(; power::Option{<:Integer} = nothing,
                  alg::AbstractDistanceAlgorithm = SimpleDistance())::Distance
    return Distance(power, alg)
end
"""
    const RhoDistanceAlgorithm = Union{SimpleDistance, SimpleAbsoluteDistance,
                                       LogDistance, CorrelationDistance}

Union of the correlation-based distance algorithms: those whose distance matrix is a pure function of a correlation matrix via [`_dist_from_cor`](@ref). Excludes [`VariationInfoDistance`](@ref) (information-theoretic, computed from the data matrix) and [`CanonicalDistance`](@ref) (a redirect that selects one of the others from the covariance estimator).

# Related

  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`distance`](@ref)
"""
const RhoDistanceAlgorithm = Union{SimpleDistance, SimpleAbsoluteDistance, LogDistance,
                                   CorrelationDistance}
"""
    _absguard(rho::MatNum)

Return `rho` unchanged if every entry is non-negative, otherwise `abs.(rho)`. Shared by [`SimpleAbsoluteDistance`](@ref) and [`LogDistance`](@ref), which are defined on the magnitude of the correlation.
"""
function _absguard(rho::MatNum)
    return all(x -> zero(x) <= x, rho) ? rho : abs.(rho)
end
"""
    _as_correlation(rho::MatNum, sym::Symbol = :rho)

Validate that `rho` is square and coerce it to a correlation matrix: if the diagonal is not all ones it is treated as a covariance matrix and converted via `StatsBase.cov2cor`. The square-matrix check runs here, once, for every correlation-based algorithm's matrix entry point.
"""
function _as_correlation(rho::MatNum, sym::Symbol = :rho)
    assert_matrix_issquare(rho, sym)
    s = LinearAlgebra.diag(rho)
    if any(!isone, s)
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return rho
end
"""
    _dist_from_cor(alg::RhoDistanceAlgorithm, power, rho::MatNum)

Turn a correlation matrix `rho` into a distance matrix for a correlation-based algorithm (see [`RhoDistanceAlgorithm`](@ref)). This is the shared kernel behind the [`distance`](@ref) and [`cor_and_dist`](@ref) entry points: they differ only in how they obtain `rho`. `power` is `nothing` for the base distance or an `Integer` for the generalised power distance; dispatch keeps the two apart so the base case never raises `rho` to a power.
"""
function _dist_from_cor(::SimpleDistance, ::Nothing, rho::MatNum)
    return sqrt.(clamp!((one(eltype(rho)) .- rho) * 0.5, zero(eltype(rho)),
                        one(eltype(rho))))
end
function _dist_from_cor(::SimpleDistance, power::Integer, rho::MatNum)
    scale = isodd(power) ? 0.5 : 1.0
    return sqrt.(clamp!((one(eltype(rho)) .- rho .^ power) * scale, zero(eltype(rho)),
                        one(eltype(rho))))
end
function _dist_from_cor(::SimpleAbsoluteDistance, ::Nothing, rho::MatNum)
    rho = _absguard(rho)
    return sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
end
function _dist_from_cor(::SimpleAbsoluteDistance, power::Integer, rho::MatNum)
    rho = _absguard(rho)
    return sqrt.(clamp!(one(eltype(rho)) .- rho .^ power, zero(eltype(rho)),
                        one(eltype(rho))))
end
function _dist_from_cor(::LogDistance, ::Nothing, rho::MatNum)
    return -log.(_absguard(rho))
end
function _dist_from_cor(::LogDistance, power::Integer, rho::MatNum)
    return -log.(_absguard(rho) .^ power)
end
function _dist_from_cor(::CorrelationDistance, ::Nothing, rho::MatNum)
    return sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
end
function _dist_from_cor(::CorrelationDistance, power::Integer, rho::MatNum)
    return sqrt.(clamp!(one(eltype(rho)) .- rho .^ power, zero(eltype(rho)),
                        one(eltype(rho))))
end
"""
    distance(de::Distance{<:Any,
                          <:Union{<:SimpleDistance, <:SimpleAbsoluteDistance, <:LogDistance,
                                  <:CorrelationDistance, <:CanonicalDistance}},
             ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

This method computes the correlation matrix using the provided covariance estimator `ce` and data matrix `X`, which is used to compute the distance matrix based on the specified distance algorithm in `de`.

# Arguments

  - `de`: Distance estimator.

      + `de::Distance{<:Any, <:SimpleDistance}`: Use the [`SimpleDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:SimpleAbsoluteDistance}`: Use the [`SimpleAbsoluteDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:LogDistance}`: Use the [`LogDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CorrelationDistance}`: Use the [`CorrelationDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CanonicalDistance}`: Use the [`CanonicalDistance`](@ref) algorithm.

  - `ce`: Covariance estimator.

  - `X`: Data matrix (observations × features).

  - $(arg_dict[:dims])

  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `D::Matrix{<:Number}`: Matrix of pairwise distances.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(de::Distance{<:Any, <:RhoDistanceAlgorithm},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return _dist_from_cor(de.alg, de.power, Statistics.cor(ce, X; dims = dims, kwargs...))
end
function distance(de::Distance{<:Any, <:CanonicalDistance},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power, alg = SimpleDistance()), ce, X;
                    dims = dims, kwargs...)
end
"""
    const LTDCov_AllInternalLTDCov = Union{<:LowerTailDependenceCovariance,
                                           <:PortfolioOptimisersCovariance{<:LowerTailDependenceCovariance}}

Alias for all internal lower tail dependence covariance estimator types.

Matches [`LowerTailDependenceCovariance`](@ref) or any [`PortfolioOptimisersCovariance`](@ref) wrapping it. Used internally for dispatch in distance computation.

# Related

  - [`LowerTailDependenceCovariance`](@ref)
  - [`DistCov_AllInternalDistCov`](@ref)
"""
const LTDCov_AllInternalLTDCov = Union{<:LowerTailDependenceCovariance,
                                       <:PortfolioOptimisersCovariance{<:LowerTailDependenceCovariance}}
"""
    distance(de::Distance{<:Any, <:LogDistance},
             ce::LTDCov_AllInternalLTDCov,
             X::MatNum; dims::Int = 1, kwargs...)

Compute the log-distance matrix from a Lower Tail Dependence (LTD) covariance estimator and data matrix.

# Arguments

  - `de::Distance{<:Any, <:LogDistance}`: Distance estimator with [`LogDistance`](@ref) algorithm.
  - `ce`: LTD covariance estimator or a PortfolioOptimisersCovariance wrapping an LTD estimator.
  - `X`: Data matrix (observations × features).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `D::Matrix{<:Number}`: Matrix of pairwise log-distances.

# Related

  - [`Distance`](@ref)
  - [`LogDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(::Distance{Nothing, <:LogDistance}, ce::LTDCov_AllInternalLTDCov,
                  X::MatNum; dims::Int = 1, kwargs...)
    rho = Statistics.cor(ce, X; dims = dims, kwargs...)
    return -log.(rho)
end
function distance(de::Distance{<:Integer, <:LogDistance}, ce::LTDCov_AllInternalLTDCov,
                  X::MatNum; dims::Int = 1, kwargs...)
    rho = Statistics.cor(ce, X; dims = dims, kwargs...)
    return -log.(rho .^ de.power)
end
"""
    distance(de::Distance{<:Any, <:VariationInfoDistance}, ::Any, X::MatNum;
             dims::Int = 1, kwargs...)

Compute the variation of information (VI) distance matrix from a data matrix.

# Arguments

  - `de::Distance{<:Any, <:VariationInfoDistance}`: Distance estimator with [`VariationInfoDistance`](@ref) algorithm.
  - `::Any`: Covariance estimator placeholder for API compatibility (ignored).
  - `X`: Data matrix (observations × features).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `D::Matrix{<:Number}`: Matrix of pairwise variation of information distances.

# Details

  - The number of bins and normalisation are taken from the [`VariationInfoDistance`](@ref) algorithm fields.
  - If `dims == 2`, the data matrix is transposed before computation.

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
"""
function distance(de::Distance{Nothing, <:VariationInfoDistance}, ::Any, X::MatNum;
                  dims::Int = 1, kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise)
end
function distance(de::Distance{<:Integer, <:VariationInfoDistance}, ::Any, X::MatNum;
                  dims::Int = 1, kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise) .^ de.power
end
"""
    distance(::Distance{<:Any,
                        <:Union{<:SimpleDistance, <:SimpleAbsoluteDistance, <:LogDistance,
                                <:CorrelationDistance, <:CanonicalDistance}},
             rho::MatNum, args...; kwargs...)

Compute the distance matrix from a correlation or covariance matrix.

If the input `rho` is a covariance matrix, it is converted to a correlation matrix which is used to compute the distance matrix using the specified distance algorithm in `de`.

# Arguments

  - `de`: Distance estimator.

      + `de::Distance{<:Any, <:SimpleDistance}`: Use the [`SimpleDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:SimpleAbsoluteDistance}`: Use the [`SimpleAbsoluteDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:LogDistance}`: Use the [`LogDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CorrelationDistance}`: Use the [`CorrelationDistance`](@ref) algorithm.
      + `de::Distance{<:Any, <:CanonicalDistance}`: Use the [`CanonicalDistance`](@ref) algorithm.

  - `rho`: Correlation or covariance matrix.

  - `args...`: Additional arguments (ignored).

  - `kwargs...`: Additional keyword arguments.

# Returns

  - `D::Matrix{<:Number}`: Matrix of pairwise Euclidean distances.

# Details

  - If `rho` is a covariance matrix, it is converted to a correlation matrix using `StatsBase.cov2cor`.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(de::Distance{<:Any, <:RhoDistanceAlgorithm}, rho::MatNum, args...;
                  kwargs...)
    return _dist_from_cor(de.alg, de.power, _as_correlation(rho))
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, rho::MatNum, args...; kwargs...)
    return distance(Distance(; power = de.power, alg = SimpleDistance()), rho; kwargs...)
end
"""
    cor_and_dist(de::Distance{<:Any,
                              <:Union{<:SimpleDistance, <:SimpleAbsoluteDistance, <:LogDistance,
                                      <:LogDistance, <:VariationInfoDistance,
                                      <:CorrelationDistance}},
                 ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute and return the correlation and distance matrices. The distance matrix depends on the combination of distance and covariance estimators (see [`distance`](@ref)).

# Arguments

  - `de`: Distance estimator.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `rho::Matrix{<:Number}`: Correlation matrix.
  - `D::Matrix{<:Number}`: Distance matrix.

# Related

  - [`Distance`](@ref)
  - [`distance`](@ref)
"""
function cor_and_dist(de::Distance{<:Any, <:RhoDistanceAlgorithm},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    rho = Statistics.cor(ce, X; dims = dims, kwargs...)
    return rho, _dist_from_cor(de.alg, de.power, rho)
end
function cor_and_dist(::Distance{Nothing, <:LogDistance}, ce::LTDCov_AllInternalLTDCov,
                      X::MatNum; dims::Int = 1, kwargs...)
    rho = Statistics.cor(ce, X; dims = dims, kwargs...)
    return rho, -log.(rho)
end
function cor_and_dist(de::Distance{<:Integer, <:LogDistance}, ce::LTDCov_AllInternalLTDCov,
                      X::MatNum; dims::Int = 1, kwargs...)
    rho = Statistics.cor(ce, X; dims = dims, kwargs...)
    return rho, -log.(rho .^ de.power)
end
function cor_and_dist(de::Distance{<:Any, <:VariationInfoDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    assert_dims(dims)
    rho = Statistics.cor(ce, X; dims = dims, kwargs...)
    return rho, distance(de, ce, X; dims = dims, kwargs...)
end
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                      X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power,
                                 alg = VariationInfoDistance(; bins = ce.bins,
                                                             normalise = ce.normalise)), ce,
                        X; dims = dims, kwargs...)
end
"""
    const AllInternalMutualInfoCov = Union{<:PortfolioOptimisersCovariance{<:MutualInfoCovariance}}

Alias for all internal mutual information covariance wrapper types.

Matches any [`PortfolioOptimisersCovariance`](@ref) wrapping a [`MutualInfoCovariance`](@ref). Used internally for dispatch in canonical distance computation.

# Related

  - [`MutualInfoCovariance`](@ref)
  - [`DistCov_AllInternalDistCov`](@ref)
"""
const AllInternalMutualInfoCov = Union{<:PortfolioOptimisersCovariance{<:MutualInfoCovariance}}
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance},
                      ce::AllInternalMutualInfoCov, X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power,
                                 alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                             normalise = ce.ce.normalise)),
                        ce, X; dims = dims, kwargs...)
end
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance},
                      ce::LTDCov_AllInternalLTDCov, X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power, alg = LogDistance()), ce, X;
                        dims = dims, kwargs...)
end
"""
    const DistCov_AllInternalDistCov = Union{<:DistanceCovariance,
                                             <:PortfolioOptimisersCovariance{<:DistanceCovariance}}

Alias for all internal distance covariance estimator types.

Matches [`DistanceCovariance`](@ref) or any [`PortfolioOptimisersCovariance`](@ref) wrapping it. Used internally for dispatch in canonical distance computation.

# Related

  - [`DistanceCovariance`](@ref)
  - [`LTDCov_AllInternalLTDCov`](@ref)
"""
const DistCov_AllInternalDistCov = Union{<:DistanceCovariance,
                                         <:PortfolioOptimisersCovariance{<:DistanceCovariance}}
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance},
                      ce::DistCov_AllInternalDistCov, X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power, alg = CorrelationDistance()), ce, X;
                        dims = dims, kwargs...)
end
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    return cor_and_dist(Distance(; power = de.power, alg = SimpleDistance()), ce, X;
                        dims = dims, kwargs...)
end
"""
    distance(de::Distance{<:Any, <:CanonicalDistance},
             ce::Union{<:MutualInfoCovariance,
                       <:AllInternalMutualInfoCov,
                       <:LTDCov_AllInternalLTDCov,
                       <:DistCov_AllInternalDistCov},
             X::MatNum; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using the covariance estimator and data matrix. The method selects the appropriate distance algorithm based on the type of covariance estimator provided (see [`CanonicalDistance`](@ref)).

# Arguments

  - `de::Distance{<:Any, <:CanonicalDistance}`: Distance estimator using the [`CanonicalDistance`](@ref) algorithm.
  - `ce::MutualInfoCovariance`: Mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `D::Matrix{<:Number}`: Matrix of pairwise canonical distances.

# Related

  - [`Distance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
  - [`LowerTailDependenceCovariance`](@ref)
  - [`DistanceCovariance`](@ref)
"""
function distance(de::Distance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power,
                             alg = VariationInfoDistance(; bins = ce.bins,
                                                         normalise = ce.normalise)), ce, X;
                    dims = dims, kwargs...)
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, ce::AllInternalMutualInfoCov,
                  X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power,
                             alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                         normalise = ce.ce.normalise)), ce,
                    X; dims = dims, kwargs...)
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, ce::LTDCov_AllInternalLTDCov,
                  X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power, alg = LogDistance()), ce, X; dims = dims,
                    kwargs...)
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, ce::DistCov_AllInternalDistCov,
                  X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power, alg = CorrelationDistance()), ce, X;
                    dims = dims, kwargs...)
end

export Distance, distance, cor_and_dist
