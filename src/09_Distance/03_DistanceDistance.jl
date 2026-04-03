"""
$(DocStringExtensions.TYPEDEF)

Distance-of-distances estimator for portfolio optimization.

`DistanceDistance` wraps a distance metric from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl) and a distance algorithm, allowing you to compute a "distance of distances" matrix. If `power` is not `nothing`, it computes the generalised distance matrix, which is then used to compute the distances of distances matrix.

```math
\\begin{align}
    _{g}\\tilde{d}_{i,\\,j} &= \\lVert_{g}\\bm{D}_{i} - _{g}\\bm{D}_{j}\\rVert\\,,
\\end{align}
```

where ``_{g}\\tilde{d}`` is the general distance of distances, ``_{g}\\bm{D}_{i}`` is the row corresponding to asset ``i`` of the general distance matrix computed using the specified distance algorithm [`AbstractDistanceAlgorithm`](@ref), ``\\lVert \\cdot \\rVert`` is the metric used to compute the distance of distances.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistanceDistance(;
        metric::Distances.Metric = Distances.Euclidean(),
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        power::Option{<:Integer} = 1,
        alg::AbstractDistanceAlgorithm = SimpleDistance()
    ) -> DistanceDistance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:dpower])

# Examples

```jldoctest
julia> DistanceDistance()
DistanceDistance
    metric ┼ Distances.Euclidean: Distances.Euclidean(0.0)
      args ┼ Tuple{}: ()
    kwargs ┼ @NamedTuple{}: NamedTuple()
   power ┼ nothing
     alg ┴ SimpleDistance()
```

# Related

  - [`DistanceDistance`](@ref)
  - [`distance`](@ref)
  - [`Distances.jl`](https://github.com/JuliaStats/Distances.jl)
"""
@concrete struct DistanceDistance <: AbstractDistanceEstimator
    "$(field_dict[:dmetric])"
    metric
    "$(field_dict[:dmetric_args])"
    args
    "$(field_dict[:dmetric_kwargs])"
    kwargs
    "$(field_dict[:dpower])"
    power
    "$(field_dict[:dalg])"
    alg
    function DistanceDistance(metric::Distances.Metric, args::Tuple, kwargs::NamedTuple,
                              power::Option{<:Integer}, alg::AbstractDistanceAlgorithm)
        if !isnothing(power)
            @argcheck(one(power) <= power, DomainError)
        end
        return new{typeof(metric), typeof(args), typeof(kwargs), typeof(power),
                   typeof(alg)}(metric, args, kwargs, power, alg)
    end
end
function DistanceDistance(; metric::Distances.Metric = Distances.Euclidean(),
                          args::Tuple = (), kwargs::NamedTuple = (;),
                          power::Option{<:Integer} = nothing,
                          alg::AbstractDistanceAlgorithm = SimpleDistance())
    return DistanceDistance(metric, args, kwargs, power, alg)
end
"""
    distance(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::MatNum;
             dims::Int = 1, kwargs...)

Compute the distance-of-distances matrix from a covariance estimator and data matrix.

This method first computes a base distance matrix using [`Distance`](@ref) with the specified power and algorithm, then applies the provided metric to compute a second-level distance matrix.

# Arguments

  - `de`: Distance-of-distances estimator.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the base distance computation.

# Returns

  - `D::Matrix{<:Number}`: Matrix of pairwise distances of distances.

# Related

  - [`DistanceDistance`](@ref)
  - [`Distance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::MatNum;
                  dims::Int = 1, kwargs...)
    D = distance(Distance(; power = de.power, alg = de.alg), ce, X; dims = dims, kwargs...)
    return Distances.pairwise(de.metric, D, de.args...; de.kwargs...)
end
"""
    distance(de::DistanceDistance, rho::MatNum, args...; kwargs...)

Compute the distance-of-distances matrix from a correlation or covariance matrix.

This method first computes a base distance matrix using [`Distance`](@ref) with the specified power and algorithm, then applies the provided metric to compute a second-level distance matrix.

# Arguments

  - `de`: Distance-of-distances estimator.
  - `rho`: Correlation or covariance matrix.
  - `args...`: Additional arguments (ignored).
  - `kwargs...`: Additional keyword arguments passed to the base distance computation.

# Returns

  - `D::Matrix{<:Number}`: Matrix of pairwise distances of distances.

# Related

  - [`DistanceDistance`](@ref)
  - [`Distance`](@ref)
  - [`distance`](@ref)

```
```
"""
function distance(de::DistanceDistance, rho::MatNum, args...; kwargs...)
    D = distance(Distance(; power = de.power, alg = de.alg), rho, args...; kwargs...)
    return Distances.pairwise(de.metric, D, de.args...; de.kwargs...)
end
"""
    cor_and_dist(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::MatNum;
                 dims::Int = 1, kwargs...)

Compute both the correlation matrix and the distance-of-distances matrix from a covariance estimator and data matrix.

This method first computes the correlation and base distance matrices using [`Distance`](@ref), then applies the provided metric to the base distance matrix.

# Arguments

  - `de`: Distance-of-distances estimator.
  - `ce`: Covariance estimator.
  - `X`: Data matrix (observations × features).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the base distance computation.

# Returns

  - `(rho::Matrix{<:Number}, D::Matrix{<:Number})`: Tuple of correlation matrix and distance-of-distances matrix.

# Related

  - [`DistanceDistance`](@ref)
  - [`Distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function cor_and_dist(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::MatNum;
                      dims::Int = 1, kwargs...)
    rho, D = cor_and_dist(Distance(; power = de.power, alg = de.alg), ce, X; dims = dims,
                          kwargs...)
    return rho, Distances.pairwise(de.metric, D, de.args...; de.kwargs...)
end

export DistanceDistance
