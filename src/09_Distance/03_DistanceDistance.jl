"""
$(DocStringExtensions.TYPEDEF)

Measures how differently two assets relate to the whole universe, by applying a metric to a distance matrix.

Two assets are close under this estimator when their *columns* of the base distance matrix are close — that is, when they stand at similar distances from every other asset. This is a second-order reading: it can separate two assets that are equally far apart under [`Distance`](@ref) but occupy different positions in the universe. It wraps a metric from [`Distances.jl`](https://github.com/JuliaStats/Distances.jl) around a base [`Distance`](@ref) built from `power` and `alg`.

!!! note

    `power = 1` reproduces the base distance exactly, so the distance-of-distances matrix is the same as at `power = nothing`. Only ``p \\geq 2`` changes the result. See [`Distance`](@ref) for the formula of each algorithm.

# Mathematical definition

```math
\\begin{align}
    _{g}\\tilde{d}_{i,\\,j} &= \\lVert_{g}\\boldsymbol{D}_{i} - _{g}\\boldsymbol{D}_{j}\\rVert\\,,
\\end{align}
```

Where:

  - ``_{g}\\tilde{d}_{i,\\,j}``: General distance of distances between assets ``i`` and ``j``.
  - ``_{g}\\boldsymbol{D}_{i}``: Column ``i`` of the generalised distance matrix (see [`AbstractDistanceAlgorithm`](@ref)).
  - ``\\lVert \\cdot \\rVert``: Metric used to compute the distance of distances, `metric`.

The source states this at the default `metric`, the Euclidean norm. A base distance matrix is symmetric, so the column and the row give the same answer.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistanceDistance(;
        metric::Distances.Metric = Distances.Euclidean(),
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        power::Option{<:Integer} = nothing,
        alg::AbstractDistanceAlgorithm = SimpleDistance()
    ) -> DistanceDistance

Keywords correspond to the struct's fields. `power` and `alg` are forwarded to a [`Distance`](@ref), so they carry the meaning and the defaults documented there.

## Validation

  - $(val_dict[:dopower])

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

!!! warning "The default metric leaves the unit interval"

    The Euclidean norm of two columns of a bounded distance matrix is not itself bounded by `1`. On a 6-asset sample two thirds of the off-diagonal entries exceeded `1`, and the largest was `1.3946164799008962`.

    [`ComplementSimilarity`](@ref) is therefore out of domain against this estimator's own default, and [`assert_similarity_domain`](@ref) refuses the pair on the PMFG path. Use [`ExponentialSimilarity`](@ref) or [`GeneralExponentialSimilarity`](@ref), which have no domain.

# Related

  - [`AbstractDistanceEstimator`](@ref)
  - [`Distance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`assert_similarity_domain`](@ref)
  - [`Distances.jl`](https://github.com/JuliaStats/Distances.jl)

# References

  - $(ref_dict[:cajas2025]) Section 12.1.1, Equation 12.1.
"""
@concrete struct DistanceDistance <: AbstractDistanceEstimator
    """
    $(field_dict[:dmetric])
    """
    metric
    """
    $(field_dict[:dmetric_args])
    """
    args
    """
    $(field_dict[:dmetric_kwargs])
    """
    kwargs
    """
    $(field_dict[:dopower])
    """
    power
    """
    $(field_dict[:dalg])
    """
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
                          alg::AbstractDistanceAlgorithm = SimpleDistance())::DistanceDistance
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
  - `X`: Data matrix (observations × assets).
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
  - `X`: Data matrix (observations × assets).
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
