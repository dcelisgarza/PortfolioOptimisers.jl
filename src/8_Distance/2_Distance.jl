"""
```julia
struct Distance{T1} <: AbstractDistanceEstimator
    alg::T1
end
```

Distance estimator for portfolio optimization.

# Fields

  - `alg`: The distance algorithm.

# Constructor

```julia
Distance(; alg::AbstractDistanceAlgorithm = SimpleDistance())
```

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> Distance()
Distance
  alg | SimpleDistance()
```

# Related

  - [`AbstractDistanceEstimator`](@ref)
  - [`AbstractDistanceAlgorithm`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`distance`](@ref)
"""
struct Distance{T1} <: AbstractDistanceEstimator
    alg::T1
    function Distance(alg::AbstractDistanceAlgorithm)
        return new{typeof(alg)}(alg)
    end
end
function Distance(; alg::AbstractDistanceAlgorithm = SimpleDistance())
    return Distance(alg)
end

"""
```julia
distance(de::Distance{<:Union{<:SimpleDistance, <:SimpleAbsoluteDistance, <:LogDistance,
                              <:CorrelationDistance, <:CanonicalDistance}},
         ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1, kwargs...)
```

This method computes the correlation matrix using the provided covariance estimator `ce` and data matrix `X`, which is used to compute the distance matrix based on the specified distance algorithm in `de`.

# Arguments

  - `de`: Distance estimator.

      + `de::Distance{<:SimpleDistance}`: Use the [`SimpleDistance`](@ref) algorithm.
      + `de::Distance{<:SimpleAbsoluteDistance}`: Use the [`SimpleAbsoluteDistance`](@ref) algorithm.
      + `de::Distance{<:LogDistance}`: Use the [`LogDistance`](@ref) algorithm.
      + `de::Distance{<:CorrelationDistance}`: Use the [`CorrelationDistance`](@ref) algorithm.
      + `de::Distance{<:CanonicalDistance}`: Use the [`CanonicalDistance`](@ref) algorithm.

  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise distances.

# Related

  - [`Distance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
end
function distance(::Distance{<:SimpleAbsoluteDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!((one(eltype(X)) .- (all(x -> x >= zero(x), rho) ? rho : abs.(rho))),
                        zero(eltype(X)), one(eltype(X))))
end
function distance(::Distance{<:LogDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return -log.(all(x -> x >= zero(x), rho) ? rho : abs.(rho))
end
function distance(::Distance{<:CorrelationDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function distance(::Distance{<:CanonicalDistance}, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(Distance(; alg = SimpleDistance()), ce, X; dims = dims, kwargs...)
end

"""
```julia
distance(de::Distance{<:LogDistance},
         ce::Union{<:LTDCovariance,
                   <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
         X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the log-distance matrix from a Lower Tail Dependence (LTD) covariance estimator and data matrix.

# Arguments

  - `::Distance{<:LogDistance}`: Distance estimator with [`LogDistance`](@ref) algorithm.
  - `ce`: LTD covariance estimator or a PortfolioOptimisersCovariance wrapping an LTD estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise log-distances.

# Related

  - [`Distance`](@ref)
  - [`LogDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function distance(::Distance{<:LogDistance},
                  ce::Union{<:LTDCovariance,
                            <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return -log.(rho)
end

"""
```julia
distance(::Distance{<:CanonicalDistance},
         ce::Union{<:MutualInfoCovariance,
                   <:PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                   <:LTDCovariance, <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any},
                   <:DistanceCovariance,
                   <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
         X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the canonical distance matrix using the covariance estimator and data matrix. The method selects the appropriate distance algorithm based on the type of covariance estimator provided (see [`CanonicalDistance`](@ref)).

# Arguments

  - `::Distance{<:CanonicalDistance}`: Distance estimator using the [`CanonicalDistance`](@ref) algorithm.
  - `ce::MutualInfoCovariance`: Mutual information covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise canonical distances.

# Related

  - [`Distance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`MutualInfoCovariance`](@ref)
  - [`LTDCovariance`](@ref)
  - [`DistanceCovariance`](@ref)
"""
function distance(::Distance{<:CanonicalDistance}, ce::MutualInfoCovariance,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(Distance(;
                             alg = VariationInfoDistance(; bins = ce.bins,
                                                         normalise = ce.normalise)), ce, X;
                    dims = dims, kwargs...)
end
function distance(::Distance{<:CanonicalDistance},
                  ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(Distance(;
                             alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                         normalise = ce.ce.normalise)), ce,
                    X; dims = dims, kwargs...)
end
function distance(::Distance{<:CanonicalDistance},
                  ce::Union{<:LTDCovariance,
                            <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(Distance(; alg = LogDistance()), ce, X; dims = dims, kwargs...)
end
function distance(::Distance{<:CanonicalDistance},
                  ce::Union{<:DistanceCovariance,
                            <:PortfolioOptimisersCovariance{<:DistanceCovariance, <:Any}},
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    return distance(Distance(; alg = CorrelationDistance()), ce, X; dims = dims, kwargs...)
end

"""
```julia
distance(de::Distance{<:VariationInfoDistance}, ::Any, X::AbstractMatrix; dims::Int = 1,
         kwargs...)
```

Compute the variation of information (VI) distance matrix from a data matrix.

# Arguments

  - `de::Distance{<:VariationInfoDistance}`: Distance estimator with [`VariationInfoDistance`](@ref) algorithm.
  - `::Any`: Covariance estimator placeholder for API compatibility (ignored).
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the distance. If `2`, the data is transposed.
  - `kwargs...`: Additional keyword arguments (ignored).

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise variation of information distances.

# Details

  - The number of bins and normalisation are taken from the [`VariationInfoDistance`](@ref) algorithm fields.
  - If `dims == 2`, the data matrix is transposed before computation.

# Related

  - [`Distance`](@ref)
  - [`VariationInfoDistance`](@ref)
"""
function distance(de::Distance{<:VariationInfoDistance}, ::Any, X::AbstractMatrix;
                  dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return variation_info(X, de.alg.bins, de.alg.normalise)
end

"""
```julia
distance(::Distance{<:Union{<:SimpleDistance, <:SimpleAbsoluteDistance, <:LogDistance,
                            <:CorrelationDistance, <:CanonicalDistance}},
         rho::AbstractMatrix, args...; kwargs...)
```

Compute the distance matrix from a correlation or covariance matrix.

If the input `rho` is a covariance matrix, it is converted to a correlation matrix which is used to compute the distance matrix using the specified distance algorithm in `de`.

# Arguments

  - `de`: Distance estimator.

      + `de::Distance{<:SimpleDistance}`: Use the [`SimpleDistance`](@ref) algorithm.
      + `de::Distance{<:SimpleAbsoluteDistance}`: Use the [`SimpleAbsoluteDistance`](@ref) algorithm.
      + `de::Distance{<:LogDistance}`: Use the [`LogDistance`](@ref) algorithm.
      + `de::Distance{<:CorrelationDistance}`: Use the [`CorrelationDistance`](@ref) algorithm.
      + `de::Distance{<:CanonicalDistance}`: Use the [`CanonicalDistance`](@ref) algorithm.

  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise Euclidean distances.

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
function distance(::Distance{<:SimpleDistance}, rho::AbstractMatrix, args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!((one(eltype(rho)) .- rho) * 0.5, zero(eltype(rho)),
                        one(eltype(rho))))
end
function distance(::Distance{<:SimpleAbsoluteDistance}, rho::AbstractMatrix, args...;
                  kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- (all(x -> x >= zero(x), rho) ? rho : abs.(rho)),
                        zero(eltype(rho)), one(eltype(rho))))
end
function distance(::Distance{<:LogDistance}, rho::AbstractMatrix, args...; kwargs...)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return -log.(all(x -> x >= zero(x), rho) ? rho : abs.(rho))
end
function distance(::Distance{<:CorrelationDistance}, rho::AbstractMatrix, args...;
                  kwargs...)
    assert_matrix_issquare(rho)
    s = diag(rho)
    iscov = any(!isone, s)
    if iscov
        s .= sqrt.(s)
        rho = StatsBase.cov2cor(rho, s)
    end
    return sqrt.(clamp!(one(eltype(rho)) .- rho, zero(eltype(rho)), one(eltype(rho))))
end
function distance(::Distance{<:CanonicalDistance}, rho::AbstractMatrix, args...; kwargs...)
    return distance(Distance(; alg = SimpleDistance()), rho; kwargs...)
end

"""
```julia
cor_and_dist(de::Distance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
             dims::Int = 1, kwargs...)
```

Compute and return the correlation and distance matrices. The distance matrix depends on the combination of distance and covariance estimators (see [`distance`](@ref)).

# Arguments

  - `de`: Distance estimator.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the correlation computation.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and distance matrix.

# Related

  - [`Distance`](@ref)
  - [`distance`](@ref)
"""
function cor_and_dist(::Distance{<:SimpleDistance}, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho,
           sqrt.(clamp!((one(eltype(X)) .- rho) * 0.5, zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(::Distance{<:SimpleAbsoluteDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho,
           sqrt.(clamp!((one(eltype(X)) .- (all(x -> x >= zero(x), rho) ? rho : abs.(rho))),
                        zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(::Distance{<:LogDistance}, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho, -log.(all(x -> x >= zero(x), rho) ? rho : abs.(rho))
end
function cor_and_dist(::Distance{<:LogDistance},
                      ce::Union{<:LTDCovariance,
                                <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho, -log.(rho)
end
function cor_and_dist(de::Distance{<:VariationInfoDistance},
                      ce::StatsBase.CovarianceEstimator, X::AbstractMatrix; dims::Int = 1,
                      kwargs...)
    @argcheck(dims in (1, 2))
    rho = cor(ce, X; dims = dims, kwargs...)
    if dims == 2
        X = transpose(X)
    end
    return rho, variation_info(X, de.alg.bins, de.alg.normalise)
end
function cor_and_dist(::Distance{<:CorrelationDistance}, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho = cor(ce, X; dims = dims, kwargs...)
    return rho, sqrt.(clamp!(one(eltype(X)) .- rho, zero(eltype(X)), one(eltype(X))))
end
function cor_and_dist(::Distance{<:CanonicalDistance}, ce::MutualInfoCovariance,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(;
                                 alg = VariationInfoDistance(; bins = ce.bins,
                                                             normalise = ce.normalise)), ce,
                        X; dims = dims, kwargs...)
end
function cor_and_dist(::Distance{<:CanonicalDistance},
                      ce::PortfolioOptimisersCovariance{<:MutualInfoCovariance, <:Any},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(;
                                 alg = VariationInfoDistance(; bins = ce.ce.bins,
                                                             normalise = ce.ce.normalise)),
                        ce, X; dims = dims, kwargs...)
end
function cor_and_dist(::Distance{<:CanonicalDistance},
                      ce::Union{<:LTDCovariance,
                                <:PortfolioOptimisersCovariance{<:LTDCovariance, <:Any}},
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; alg = LogDistance()), ce, X; dims = dims, kwargs...)
end
function cor_and_dist(::Distance{<:CanonicalDistance},
                      ce::Union{<:DistanceCovariance,
                                <:PortfolioOptimisersCovariance{<:DistanceCovariance,
                                                                <:Any}}, X::AbstractMatrix;
                      dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; alg = CorrelationDistance()), ce, X; dims = dims,
                        kwargs...)
end
function cor_and_dist(::Distance{<:CanonicalDistance}, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    return cor_and_dist(Distance(; alg = SimpleDistance()), ce, X; dims = dims, kwargs...)
end

export Distance
