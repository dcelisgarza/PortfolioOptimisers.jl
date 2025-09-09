"""
    struct GeneralDistance{T1, T2} <: AbstractDistanceEstimator
        power::T1
        alg::T2
    end

Generalised distance estimator for portfolio optimization.

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

  - `power`: The integer power to which the base correlation or distance matrix is raised.
  - `alg`: The base distance algorithm.

# Constructor

    GeneralDistance(; power::Integer = 1, alg::AbstractDistanceAlgorithm = SimpleDistance())

Keyword arguments correspond to the fields above.

## Validation

  - `power >= 1`.

# Examples

```jldoctest
julia> GeneralDistance()
GeneralDistance
  power | Int64: 1
    alg | SimpleDistance()
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
struct GeneralDistance{T1, T2} <: AbstractDistanceEstimator
    power::T1
    alg::T2
end
function GeneralDistance(; power::Integer = 1,
                         alg::AbstractDistanceAlgorithm = SimpleDistance())
    @argcheck(power >= one(power))
    return GeneralDistance(power, alg)
end
"""
    distance(de::GeneralDistance{<:Any,
                                 <:Union{<:SimpleDistance, <:SimpleAbsoluteDistance,
                                         <:LogDistance, <:CorrelationDistance,
                                         <:CanonicalDistance}},
             ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
             kwargs...) 

This method computes the generalised correlation matrix using the provided covariance estimator `ce` and data matrix `X`, which is used to compute the distance matrix based on the specified distance algorithm in `de`.

# Arguments

  - `de`: General distance estimator.

      + `de::GeneralDistance{<:Any, <:SimpleDistance}`: Use the [`SimpleDistance`](@ref) algorithm.
      + `de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance}`: Use the [`SimpleAbsoluteDistance`](@ref) algorithm.
      + `de::GeneralDistance{<:Any, <:LogDistance}`: Use the [`LogDistance`](@ref) algorithm.
      + `de::GeneralDistance{<:Any, <:CorrelationDistance}`: Use the [`CorrelationDistance`](@ref) algorithm.
      + `de::GeneralDistance{<:Any, <:CanonicalDistance}`: Use the [`CanonicalDistance`](@ref) algorithm.

  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of distances of distances.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:SimpleDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
end
function distance(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    rho = (all(x -> x >= zero(x), rho) ? rho : abs.(rho)) .^ de.power
    return sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function distance(de::GeneralDistance{<:Any, <:LogDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    rho = (all(x -> x >= zero(x), rho) ? rho : abs.(rho)) .^ de.power
    return -log.(rho)
end
function distance(de::GeneralDistance{<:Any, <:CorrelationDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
                  ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                  kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = SimpleDistance()), ce, X;
                    dims = dims, kwargs...)
end
"""
    distance(de::GeneralDistance{<:Any, <:LogDistance},
             ce::Union{<:LTDCovariance,
                       <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the log-distance matrix from a Lower Tail Dependence (LTD) covariance estimator and data matrix.

# Arguments

  - `de::GeneralDistance{<:Any, <:LogDistance}`: General distance estimator with [`LogDistance`](@ref) algorithm.
  - `ce`: LTD covariance estimator or a PortfolioOptimisersCovariance wrapping an LTD estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise log-distances.

# Related

  - [`GeneralDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:LogDistance},
                  ce::Union{<:LTDCovariance,
                            <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return -log.(rho)
end
"""
    distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
             ce::Union{<:MutualInfoCovariance,
                       <:PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                       <:LTDCovariance,
                       <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any},
                       <:DistanceCovariance,
                       <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the canonical distance matrix using the covariance estimator and data matrix. The method selects the appropriate distance algorithm based on the type of covariance estimator provided (see [`CanonicalDistance`](@ref)).

# Arguments

  - `de::GeneralDistance{<:Any, <:CanonicalDistance}`: General distance estimator using the [`CanonicalDistance`](@ref) algorithm.
  - `ce::MutualInfoCovariance`: Mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`GeneralDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
  - [`LTDCovariance`](@ref)
  - [`DistanceCovariance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power,
                                    alg = VariationInfoDistance(; bins = ce.bins,
                                                                normalise = ce.normalise)),
                    ce, X; dims = dims, kwargs...)
end
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
                  ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power,
                                    alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                                normalise = ce.ce.normalise)),
                    ce, X; dims = dims, kwargs...)
end
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
                  ce::Union{<:LTDCovariance,
                            <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = LogDistance()), ce, X;
                    dims = dims, kwargs...)
end
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance},
                  ce::Union{<:DistanceCovariance,
                            <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = CorrelationDistance()), ce, X;
                    dims = dims, kwargs...)
end
"""
    distance(de::GeneralDistance{<:Any, <:VariationInfoDistance}, ::Any, 
             X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute the variation of information (VI) distance matrix from a data matrix.

# Arguments

  - `de::GeneralDistance{<:Any, <:VariationInfoDistance}`: General distance estimator with [`VariationInfoDistance`](@ref) algorithm.
  - `::Any`: Placeholder for compatibility, ignored.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance. If `2`, the data is transposed.
  - `kwargs...`: Additional keyword arguments, ignored.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise variation of information distances.

# Details

  - The number of bins and normalisation are taken from the [`VariationInfoDistance`](@ref) algorithm fields.
  - If `dims == 2`, the data matrix is transposed before computation.

# Related

  - [`GeneralDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:VariationInfoDistance}, ::Any,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise) .^ de.power
end
"""
    distance(de::GeneralDistance{<:Any,
                                 <:Union{<:SimpleDistance, <:SimpleAbsoluteDistance,
                                         <:LogDistance, <:CorrelationDistance,
                                         <:CanonicalDistance}}, rho::AbstractMatrix,
             args...; kwargs...)

Compute the general distance matrix from a correlation or covariance matrix.

If the input `rho` is a covariance matrix, it is converted to a correlation matrix which is used to compute the distance matrix using the specified distance algorithm in `de`.

# Arguments

  - `de`: General distance estimator.

      + `de::GeneralDistance{<:Any, <:SimpleDistance}`: Use the [`SimpleDistance`](@ref) algorithm.
      + `de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance}`: Use the [`SimpleAbsoluteDistance`](@ref) algorithm.
      + `de::GeneralDistance{<:Any, <:LogDistance}`: Use the [`LogDistance`](@ref) algorithm.
      + `de::GeneralDistance{<:Any, <:CorrelationDistance}`: Use the [`CorrelationDistance`](@ref) algorithm.
      + `de::GeneralDistance{<:Any, <:CanonicalDistance}`: Use the [`CanonicalDistance`](@ref) algorithm.

  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise Euclidean distances.

# Details

  - If `rho` is a covariance matrix, it is converted to a correlation matrix using `StatsBase.cov2cor`.

# Related

  - [`GeneralDistance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(de::GeneralDistance{<:Any, <:SimpleDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
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
function distance(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .-
                        (all(x -> x >= zero(x), rho) ? rho : abs.(rho)) .^ de.power,
                        zero(eltype(rho)), one(eltype(rho))))
end
function distance(de::GeneralDistance{<:Any, <:LogDistance}, rho::AbstractMatrix, args...;
                  kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.((all(x -> x >= zero(x), rho) ? rho : abs.(rho)) .^ de.power)
end
function distance(de::GeneralDistance{<:Any, <:CorrelationDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    assert_matrix_issquare(rho)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho .^ de.power, zero(eltype(rho)),
                        one(eltype(rho))))
end
function distance(de::GeneralDistance{<:Any, <:CanonicalDistance}, rho::AbstractMatrix,
                  args...; kwargs...)
    return distance(GeneralDistance(; power = de.power, alg = SimpleDistance()), rho;
                    kwargs...)
end

"""
    cor_and_dist(de::GeneralDistance, ce::StatsBase.CovarianceEstimator,
                 X::AbstractMatrix; dims::Int = 1, kwargs...)

Compute and return the correlation and distance matrices. The distance matrix depends on the combination of distance and covariance estimators (see [`distance`](@ref)).

# Arguments

  - `de`: General distance estimator.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and distance matrix.

# Related

  - [`GeneralDistance`](@ref)
  - [`distance`](@ref)
"""
function cor_and_dist(de::GeneralDistance{<:Any, <:SimpleDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    scale = isodd(de.power) ? 0.5 : 1.0
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return rho,
           sqrt.(clamp!((one(eltype(X)) .- rho) * scale, zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(de::GeneralDistance{<:Any, <:SimpleAbsoluteDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    rho = (all(x -> x >= zero(x), rho) ? rho : abs.(rho)) .^ de.power
    return rho, sqrt.(clamp!((one(eltype(X)) .- rho), zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(de::GeneralDistance{<:Any, <:LogDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    rho = (all(x -> x >= zero(x), rho) ? rho : abs.(rho)) .^ de.power
    return rho, -log.(rho)
end
function cor_and_dist(de::GeneralDistance{<:Any, <:LogDistance},
                      ce::Union{<:LTDCovariance,
                                <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return rho, -log.(rho)
end
function cor_and_dist(de::GeneralDistance{<:Any, <:VariationInfoDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    @argcheck(dims in (1, 2))
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    if dims == 2
        X = transpose(X)
    end
    return rho, variation_info(X, de.alg.bins, de.alg.normalise) .^ de.power
end
function cor_and_dist(de::GeneralDistance{<:Any, <:CorrelationDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...) .^ de.power
    return rho, sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                      ce::MutualInfoCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(GeneralDistance(; power = de.power,
                                        alg = VariationInfoDistance(; bins = ce.bins,
                                                                    normalise = ce.normalise)),
                        ce, X; dims = dims, kwargs...)
end
function cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                      ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(GeneralDistance(; power = de.power,
                                        alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                                    normalise = ce.ce.normalise)),
                        ce, X; dims = dims, kwargs...)
end
function cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                      ce::Union{<:LTDCovariance,
                                <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(GeneralDistance(; power = de.power, alg = LogDistance()), ce, X;
                        dims = dims, kwargs...)
end
function cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                      ce::Union{<:DistanceCovariance,
                                <:PortfolioOptimisersCovariance{<:DistanceCovariance,
                                                                <:Any}}, X::AbstractMatrix;
                      dims::Int = 1, kwargs...)
    return cor_and_dist(GeneralDistance(; power = de.power, alg = CorrelationDistance()),
                        ce, X; dims = dims, kwargs...)
end
function cor_and_dist(de::GeneralDistance{<:Any, <:CanonicalDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    return cor_and_dist(GeneralDistance(; power = de.power, alg = SimpleDistance()), ce, X;
                        dims = dims, kwargs...)
end

export GeneralDistance
