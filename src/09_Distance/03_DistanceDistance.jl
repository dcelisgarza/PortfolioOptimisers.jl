"""
    struct DistanceDistance{T1, T2, T3, T4, T5} <: AbstractDistanceEstimator
        dist::T1
        args::T2
        kwargs::T3
        power::T4
        alg::T5
    end

Distance-of-distances estimator for portfolio optimization.

`DistanceDistance` wraps a distance metric from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl) and a distance algorithm, allowing you to compute a "distance of distances" matrix. If `power` is not `nothing`, it computes the generalised distance matrix, which is then used to compute the distances of distances matrix.

```math
\\begin{align}
    _{g}\\tilde{d}_{i,\\,j} &= \\lVert_{g}\\bm{D}_{i} - _{g}\\bm{D}_{j}\\rVert\\,,
\\end{align}
```

where ``_{g}\\tilde{d}`` is the general distance of distances, ``_{g}\\bm{D}_{i}`` is the row corresponding to asset ``i`` of the general distance matrix computed using the specified distance algorithm [`AbstractDistanceAlgorithm`](@ref), ``\\lVert \\cdot \\rVert`` is the metric used to compute the distance of distances.

# Fields

  - `dist`: The metric to use for the second-level distance from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl).
  - `args`: Positional arguments to pass to the metric.
  - `kwargs`: Keyword arguments to pass to the metric.
  - `power`: The integer power to which the base correlation or distance matrix is raised.
  - `alg`: The base distance algorithm to use.

# Constructor

    DistanceDistance(; dist::Distances.Metric = Distances.Euclidean(), args::Tuple = (),
                     kwargs::NamedTuple = (;), power::Union{Nothing, <:Integer} = 1,
                     alg::AbstractDistanceAlgorithm = SimpleDistance())

Keyword arguments correspond to the fields above.

## Validation

  - `power >= 1`.

# Examples

```jldoctest
julia> DistanceDistance()
DistanceDistance
    dist ├ Distances.Euclidean: Distances.Euclidean(0.0)
    args ├ Tuple{}: ()
  kwargs ├ @NamedTuple{}: NamedTuple()
   power ├ nothing
     alg └ SimpleDistance()
```

# Related

  - [`DistanceDistance`](@ref)
  - [`distance`](@ref)
  - [`Distances.jl`](https://github.com/JuliaStats/Distances.jl)
"""
struct DistanceDistance{T1, T2, T3, T4, T5} <: AbstractDistanceEstimator
    dist::T1
    args::T2
    kwargs::T3
    power::T4
    alg::T5
    function DistanceDistance(dist::Distances.Metric, args::Tuple, kwargs::NamedTuple,
                              power::Union{Nothing, <:Integer},
                              alg::AbstractDistanceAlgorithm)
        if !isnothing(power)
            @argcheck(power >= one(power))
        end
        return new{typeof(dist), typeof(args), typeof(kwargs), typeof(power), typeof(alg)}(dist,
                                                                                           args,
                                                                                           kwargs,
                                                                                           power,
                                                                                           alg)
    end
end
function DistanceDistance(; dist::Distances.Metric = Distances.Euclidean(),
                          args::Tuple = (), kwargs::NamedTuple = (;),
                          power::Union{Nothing, <:Integer} = nothing,
                          alg::AbstractDistanceAlgorithm = SimpleDistance())
    return DistanceDistance(dist, args, kwargs, power, alg)
end
"""
    distance(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
             dims::Int = 1, kwargs...)

Compute the distance-of-distances matrix from a covariance estimator and data matrix.

This method first computes a base distance matrix using [`Distance`](@ref) with the specified power and algorithm, then applies the provided metric to compute a second-level distance matrix.

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
  - [`Distance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::DistanceDistance, ce::StatsBase.CovarianceEstimator,
                  X::AbstractMatrix; dims::Int = 1, kwargs...)
    dist = distance(Distance(; power = de.power, alg = de.alg), ce, X; dims = dims,
                    kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
"""
    distance(de::DistanceDistance, rho::AbstractMatrix, args...; kwargs...)

Compute the distance-of-distances matrix from a correlation or covariance matrix.

This method first computes a base distance matrix using [`Distance`](@ref) with the specified power and algorithm, then applies the provided metric to compute a second-level distance matrix.

# Arguments

  - `de`: Distance-of-distances estimator.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments passed to the base distance computation.

# Returns

  - `dist::Matrix{<:Real}`: Matrix of pairwise distances of distances.

# Related

  - [`DistanceDistance`](@ref)
  - [`Distance`](@ref)
  - [`distance`](@ref)

```
```
"""
function distance(de::DistanceDistance, rho::AbstractMatrix, args...; kwargs...)
    dist = distance(Distance(; power = de.power, alg = de.alg), rho, args...; kwargs...)
    return Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end
"""
    cor_and_dist(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::AbstractMatrix;
                 dims::Int = 1, kwargs...)

Compute both the correlation matrix and the distance-of-distances matrix from a covariance estimator and data matrix.

This method first computes the correlation and base distance matrices using [`Distance`](@ref), then applies the provided metric to the base distance matrix.

# Arguments

  - `de`: Distance-of-distances estimator.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - `dims`: Dimension along which to compute the base distance.
  - `kwargs...`: Additional keyword arguments passed to the base distance computation.

# Returns

  - `(rho::Matrix{<:Real}, D::Matrix{<:Real})`: Tuple of correlation matrix and distance-of-distances matrix.

# Related

  - [`DistanceDistance`](@ref)
  - [`Distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function cor_and_dist(de::DistanceDistance, ce::StatsBase.CovarianceEstimator,
                      X::AbstractMatrix; dims::Int = 1, kwargs...)
    rho, dist = cor_and_dist(Distance(; power = de.power, alg = de.alg), ce, X; dims = dims,
                             kwargs...)
    return rho, Distances.pairwise(de.dist, dist, de.args...; de.kwargs...)
end

export DistanceDistance
