"""
```julia
struct DistanceDistance{T1, T2, T3, T4} <: AbstractDistanceEstimator
    dist::T1
    args::T2
    kwargs::T3
    alg::T4
end
```

A distance-of-distances estimator for portfolio optimization.

`DistanceDistance` wraps a distance metric from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl) and a base distance algorithm, allowing you to compute a "distance of distances" matrix.

```math
\\begin{align}
    \\tilde{d}_{i,\\,j} &= \\lVert\\bm{D}_{i} - \\bm{D}_{j}\\rVert\\,,
\\end{align}
```

where ``\\tilde{d}`` is the distance of distances, ``\\bm{D}_{i}`` is the row corresponding to asset ``i`` of the distance matrix computed using the specified distance algorithm [`AbstractDistanceAlgorithm`](@ref), ``\\lVert \\cdot \\rVert`` is the metric used to compute the distance of distances.

# Fields

  - `dist`: The metric to use for the second-level distance from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl).
  - `args`: Positional arguments to pass to the metric.
  - `kwargs`: Keyword arguments to pass to the metric.
  - `alg::AbstractDistanceAlgorithm`: The base distance algorithm to use.

# Constructor

    DistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                       args::Tuple = (), kwargs::NamedTuple = (;),
                       alg::AbstractDistanceAlgorithm = SimpleDistance())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> DistanceDistance()
DistanceDistance
    dist | Distances.Euclidean: Distances.Euclidean(0.0)
    args | Tuple{}: ()
  kwargs | @NamedTuple{}: NamedTuple()
     alg | SimpleDistance()
```

# Related

  - [`Distance`](@ref)
  - [`distance`](@ref)
  - [`Distances.jl`](https://github.com/JuliaStats/Distances.jl)
"""
struct DistanceDistance{T1, T2, T3, T4} <: AbstractDistanceEstimator
    dist::T1
    args::T2
    kwargs::T3
    alg::T4
end
function DistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                          args::Tuple = (), kwargs::NamedTuple = (;),
                          alg::AbstractDistanceAlgorithm = SimpleDistance())
    return DistanceDistance(dist, args, kwargs, alg)
end

"""
```julia
distance(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
         dims::Int = 1, kwargs...)
```

Compute the distance-of-distances matrix from a covariance estimator and data matrix.

This method first computes a base distance matrix using the specified base distance algorithm, then applies the provided metric to compute a second-level distance matrix.

# Arguments

  - `de`: Distance-of-distances estimator.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the base distance.
  - `kwargs...`: Additional keyword arguments passed to the base distance computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise distances of distances.

# Related

  - [`DistanceDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::DistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    dist = distance(Distance(; alg = de.alg), ce, X; dims = dims, kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
"""
```julia
distance(de::DistanceDistance, rho::AbstractMatrix, args...; kwargs...)
```

Compute the distance-of-distances matrix from a correlation or covariance matrix.

This method first computes a base distance matrix using the specified base distance algorithm, then applies the provided metric to compute a second-level distance matrix.

# Arguments

  - `de`: Distance-of-distances estimator.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments passed to the base distance computation.
  - `kwargs...`: Additional keyword arguments passed to the base distance computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise distances of distances.

# Related

  - [`DistanceDistance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::DistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    dist = distance(Distance(; alg = de.alg), rho, args...; kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
"""
```julia
cor_and_dist(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
             dims::Int = 1, kwargs...)
```

Compute both the correlation matrix and the distance-of-distances matrix from a covariance estimator and data matrix.

This method first computes the correlation and base distance matrices, then applies the provided metric to the base distance matrix.

# Arguments

  - `de`: Distance-of-distances estimator.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the base distance.
  - `kwargs...`: Additional keyword arguments passed to the base distance computation.

# Returns

  - `(rho::Matrix{<:Real}, dist::Matrix{<:Real})`: Tuple of correlation matrix and distance-of-distances matrix.

# Related

  - [`DistanceDistance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function cor_and_dist(de::DistanceDistance, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho, dist = cor_and_dist(Distance(; alg = de.alg), ce, X; dims = dims, kwargs...)
    return rho, Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export DistanceDistance
