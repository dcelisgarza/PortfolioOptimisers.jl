"""
    struct Distance{T1, T2} <: AbstractDistanceEstimator
        power::T1
        alg::T2
    end

If power is not `nothing`, computes the generalised distance estimator.

```math
\\begin{align}
_{g}d_{i,\\,j} &= s \\cdot \\left(d_{i,\\,j}\\right)^{p}\\\\
    s &= \\begin{cases}
        1/2 & \\text{if } p \\mod 2 \\neq 0\\\\
        1 & \\text{otherwise}
        \\end{cases}\\,,
\\end{align}
```

where ``_{g}d`` is the generalised distance, ``d`` is the base distance computed using the specified distance algorithm, ``p`` is the integer power, ``s`` is a scaling factor, and each subscript denotes an asset.

# Fields

  - `power`: Optional integer power to which the base correlation or distance matrix is raised.
  - `alg`: The base distance algorithm.

# Constructor

    Distance(; power::Option{<:Integer} = nothing,
             alg::AbstractDistanceAlgorithm = SimpleDistance())

Keyword arguments correspond to the fields above.

## Validation

  - If `power` is not `nothing`, then `power >= 1`.

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
struct Distance{T1, T2} <: AbstractDistanceEstimator
    power::T1
    alg::T2
    function Distance(power::Option{<:Integer}, alg::AbstractDistanceAlgorithm)
        if !isnothing(power)
            @argcheck(one(power) <= power, DomainError)
        end
        return new{typeof(power), typeof(alg)}(power, alg)
    end
end
function Distance(; power::Option{<:Integer} = nothing,
                  alg::AbstractDistanceAlgorithm = SimpleDistance())
    return Distance(power, alg)
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
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Number}`: Matrix of pairwise distances.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(::Distance{Nothing, <:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                  X::MatNum; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
end
function distance(de::Distance{<:Integer, <:SimpleDistance},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
end
function distance(::Distance{Nothing, <:SimpleAbsoluteDistance},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!((one(eltype(X)) .- (all(x -> zero(x) <= x, rho) ? rho : abs.(rho))),
                        zero(eltype(X)), one(eltype(X))))
end
function distance(de::Distance{<:Integer, <:SimpleAbsoluteDistance},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    rho = (all(x -> zero(x) <= x, rho) ? rho : abs.(rho)) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function distance(::Distance{Nothing, <:LogDistance}, ce::StatsBase.CovarianceEstimator,
                  X::MatNum; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return -log.(all(x -> zero(x) <= x, rho) ? rho : abs.(rho))
end
function distance(de::Distance{<:Integer, <:LogDistance}, ce::StatsBase.CovarianceEstimator,
                  X::MatNum; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    rho = (all(x -> zero(x) <= x, rho) ? rho : abs.(rho)) .^ de.power
    return -log.(rho)
end
function distance(::Distance{Nothing, <:CorrelationDistance},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function distance(de::Distance{<:Integer, <:CorrelationDistance},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function distance(de::Distance{<:Any, <:CanonicalDistance},
                  ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power, alg = SimpleDistance()), ce, X;
                    dims = dims, kwargs...)
end
const LTDCov_PLTDCov = Union{<:LowerTailDependenceCovariance,
                             <:PortfolioOptimisersCovariance{<:LowerTailDependenceCovariance,
                                                             <:Any}}
"""
    distance(de::Distance{<:Any, <:LogDistance},
             ce::LTDCov_PLTDCov,
             X::MatNum; dims::Int = 1, kwargs...)

Compute the log-distance matrix from a Lower Tail Dependence (LTD) covariance estimator and data matrix.

# Arguments

  - `de::Distance{<:Any, <:LogDistance}`: Distance estimator with [`LogDistance`](@ref) algorithm.
  - `ce`: LTD covariance estimator or a PortfolioOptimisersCovariance wrapping an LTD estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Number}`: Matrix of pairwise log-distances.

# Related

  - [`Distance`](@ref)
  - [`LogDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(::Distance{Nothing, <:LogDistance}, ce::LTDCov_PLTDCov, X::MatNum;
                  dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return -log.(rho)
end
function distance(de::Distance{<:Integer, <:LogDistance}, ce::LTDCov_PLTDCov, X::MatNum;
                  dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return -log.(rho)
end
"""
    distance(de::Distance{<:Any, <:VariationInfoDistance}, ::Any, X::MatNum;
             dims::Int = 1, kwargs...)

Compute the variation of information (VI) distance matrix from a data matrix.

# Arguments

  - `de::Distance{<:Any, <:VariationInfoDistance}`: Distance estimator with [`VariationInfoDistance`](@ref) algorithm.
  - `::Any`: Covariance estimator placeholder for API compatibility (ignored).
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance. If `2`, the data is transposed.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `dist::Matrix{<:Number}`: Matrix of pairwise variation of information distances.

# Details

  - The number of bins and normalisation are taken from the [`VariationInfoDistance`](@ref) algorithm fields.
  - If `dims == 2`, the data matrix is transposed before computation.

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
"""
function distance(de::Distance{Nothing, <:VariationInfoDistance}, ::Any, X::MatNum;
                  dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise)
end
function distance(de::Distance{<:Integer, <:VariationInfoDistance}, ::Any, X::MatNum;
                  dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
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

  - `dist::Matrix{<:Number}`: Matrix of pairwise Euclidean distances.

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
function distance(::Distance{Nothing, <:SimpleDistance}, rho::MatNum, args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!((one(eltype(rho)) .- rho) * 0.5, zero(eltype(rho)),
                        one(eltype(rho))))
end
function distance(de::Distance{<:Integer, <:SimpleDistance}, rho::MatNum, args...;
                  kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    scale = isodd(de.power) ? 0.5 : 1.0
    return sqrt.(clamp!((one(eltype(rho)) .- rho .^ de.power) * scale, zero(eltype(rho)),
                        one(eltype(rho))))
end
function distance(::Distance{Nothing, <:SimpleAbsoluteDistance}, rho::MatNum, args...;
                  kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- (all(x -> zero(x) <= x, rho) ? rho : abs.(rho)),
                        zero(eltype(rho)), one(eltype(rho))))
end
function distance(de::Distance{<:Integer, <:SimpleAbsoluteDistance}, rho::MatNum, args...;
                  kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .-
                        (all(x -> zero(x) <= x, rho) ? rho : abs.(rho)) .^ de.power,
                        zero(eltype(rho)), one(eltype(rho))))
end
function distance(::Distance{Nothing, <:LogDistance}, rho::MatNum, args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.(all(x -> zero(x) <= x, rho) ? rho : abs.(rho))
end
function distance(de::Distance{<:Integer, <:LogDistance}, rho::MatNum, args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.((all(x -> zero(x) <= x, rho) ? rho : abs.(rho)) .^ de.power)
end
function distance(::Distance{Nothing, <:CorrelationDistance}, rho::MatNum, args...;
                  kwargs...)
    assert_matrix_issquare(rho, :rho)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
end
function distance(de::Distance{<:Integer, <:CorrelationDistance}, rho::MatNum, args...;
                  kwargs...)
    assert_matrix_issquare(rho, :rho)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho .^ de.power, zero(eltype(rho)),
                        one(eltype(rho))))
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, rho::MatNum, args...; kwargs...)
    return distance(Distance(; power = de.power, alg = SimpleDistance()), rho; kwargs...)
end
"""
    cor_and_dist(de::Distance{<:Any,
                              <:Union{<:SimpleDistance, <:SimpleAbsoluteDistance, <:LogDistance,
                                      <:LogDistance, <:VariationInfoDistance,
                                      <:CorrelationDistance}}, Nothing,
                 ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)

Compute and return the correlation and distance matrices. The distance matrix depends on the combination of distance and covariance estimators (see [`distance`](@ref)).

# Arguments

  - `de`: Distance estimator.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `(rho::Matrix{<:Number}, dist::Matrix{<:Number})`: Tuple of correlation matrix and distance matrix.

# Related

  - [`Distance`](@ref)
  - [`distance`](@ref)
"""
function cor_and_dist(::Distance{Nothing, <:SimpleDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho,
           sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(de::Distance{<:Integer, <:SimpleDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return rho,
           sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(::Distance{Nothing, <:SimpleAbsoluteDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho,
           sqrt.(clamp!((one(eltype(X)) .- (all(x -> zero(x) <= x, rho) ? rho : abs.(rho))),
                        zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(de::Distance{<:Integer, <:SimpleAbsoluteDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    rho = (all(x -> zero(x) <= x, rho) ? rho : abs.(rho)) .^ de.power
    return rho, sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(::Distance{Nothing, <:LogDistance}, ce::StatsBase.CovarianceEstimator,
                      X::MatNum; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho, -log.(all(x -> zero(x) <= x, rho) ? rho : abs.(rho))
end
function cor_and_dist(de::Distance{<:Integer, <:LogDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    rho = (all(x -> zero(x) <= x, rho) ? rho : abs.(rho)) .^ de.power
    return rho, -log.(rho)
end
function cor_and_dist(::Distance{Nothing, <:LogDistance}, ce::LTDCov_PLTDCov, X::MatNum;
                      dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho, -log.(rho)
end
function cor_and_dist(de::Distance{<:Integer, <:LogDistance}, ce::LTDCov_PLTDCov, X::MatNum;
                      dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return rho, -log.(rho)
end
function cor_and_dist(de::Distance{Nothing, <:VariationInfoDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    @argcheck(dims in (1, 2))
    rho = cor(ce, X; dims = dims, kwargs...)
    if dims == 2
        X = transpose(X)
    end
    return rho, variation_info(X, de.alg.bins, de.alg.normalise)
end
function cor_and_dist(de::Distance{<:Integer, <:VariationInfoDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    @argcheck(dims in (1, 2))
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    if dims == 2
        X = transpose(X)
    end
    return rho, variation_info(X, de.alg.bins, de.alg.normalise) .^ de.power
end
function cor_and_dist(::Distance{Nothing, <:CorrelationDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho, sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(de::Distance{<:Integer, <:CorrelationDistance},
                      ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return rho, sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                      X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power,
                                 alg = VariationInfoDistance(; bins = ce.bins,
                                                             normalise = ce.normalise)), ce,
                        X; dims = dims, kwargs...)
end
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance},
                      ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                      X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power,
                                 alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                             normalise = ce.ce.normalise)),
                        ce, X; dims = dims, kwargs...)
end
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance}, ce::LTDCov_PLTDCov,
                      X::MatNum; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; power = de.power, alg = LogDistance()), ce, X;
                        dims = dims, kwargs...)
end
const DistCov_PDistCov = Union{<:DistanceCovariance,
                               <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}}
function cor_and_dist(de::Distance{<:Any, <:CanonicalDistance}, ce::DistCov_PDistCov,
                      X::MatNum; dims::Int = 1, kwargs...)
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
                       <:PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                       <:LowerTailDependenceCovariance, <:PortfolioOptimisersCovariance{<:LowerTailDependenceCovariance, <:Any},
                       <:DistanceCovariance,
                       <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
             X::MatNum; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using the covariance estimator and data matrix. The method selects the appropriate distance algorithm based on the type of covariance estimator provided (see [`CanonicalDistance`](@ref)).

# Arguments

  - `de::Distance{<:Any, <:CanonicalDistance}`: Distance estimator using the [`CanonicalDistance`](@ref) algorithm.
  - `ce::MutualInfoCovariance`: Mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Number}`: Matrix of pairwise canonical distances.

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
function distance(de::Distance{<:Any, <:CanonicalDistance},
                  ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                  X::MatNum; dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power,
                             alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                         normalise = ce.ce.normalise)), ce,
                    X; dims = dims, kwargs...)
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, ce::LTDCov_PLTDCov, X::MatNum;
                  dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power, alg = LogDistance()), ce, X; dims = dims,
                    kwargs...)
end
function distance(de::Distance{<:Any, <:CanonicalDistance}, ce::DistCov_PDistCov, X::MatNum;
                  dims::Int = 1, kwargs...)
    return distance(Distance(; power = de.power, alg = CorrelationDistance()), ce, X;
                    dims = dims, kwargs...)
end

export Distance
