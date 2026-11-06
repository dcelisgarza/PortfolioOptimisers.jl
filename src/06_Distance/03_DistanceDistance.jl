"""
$(DocStringExtensions.TYPEDEF)

Measures how differently two assets relate to the whole universe, by a metric over the columns of a distance matrix.

Two assets are close under this estimator when their columns of the base distance matrix are close. Then the two assets have similar distances to every other asset. Thus the estimator can separate two assets that have the same distance under [`Distance`](@ref) but different positions in the universe. It applies a metric of [`Distances.jl`](https://docs.juliahub.com/General/Distances/stable/) to the matrix of a base [`Distance`](@ref), which it makes from `power` and `alg`.

!!! note

    `power = 1` gives the base distance exactly, so the distance-of-distances matrix is the same as at `power = nothing`. Only ``p \\geq 2`` changes the result. [`Distance`](@ref) states the formula of each algorithm.

# Mathematical definition

```math
\\begin{align}
    _{g}\\tilde{d}_{i,\\,j} &= \\lVert_{g}\\boldsymbol{D}_{i} - _{g}\\boldsymbol{D}_{j}\\rVert\\,,
\\end{align}
```

Where:

  - ``_{g}\\tilde{d}_{i,\\,j}``: The general distance of distances between assets ``i`` and ``j``.
  - ``_{g}\\boldsymbol{D}_{i}``: Column ``i`` of the generalised distance matrix, which [`AbstractDistanceAlgorithm`](@ref) defines.
  - ``\\lVert \\cdot \\rVert``: The metric of the distance of distances, the field `metric`.

The source states this formula with the Euclidean norm, which is the default `metric`. A base distance matrix is symmetric, so the column and the row give the same result.

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

Keywords correspond to the struct's fields. The estimator passes `power` and `alg` to a [`Distance`](@ref), so they have the meaning and the defaults that its docstring states.

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

    The Euclidean distance between two columns of a distance matrix can be larger than `1`, also when each entry of the matrix is in `[0, 1]`.

    Thus the default of this estimator can give distances outside the domain of [`ComplementSimilarity`](@ref), and [`assert_similarity_domain`](@ref) refuses that pair on the PMFG path. Use [`ExponentialSimilarity`](@ref) or [`GeneralExponentialSimilarity`](@ref), which have no domain.

# Related

  - [`AbstractDistanceEstimator`](@ref)
  - [`Distance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`assert_similarity_domain`](@ref)
  - [`Distances.jl`](https://docs.juliahub.com/General/Distances/stable/)

# References

  - $(ref_dict[:lopezdeprado2016]) Online supplementary materials, Stage 1, Example 2.
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
            @argcheck(one(power) <= power,
                      DomainError(power,
                                  "`DistanceDistance.power` is $power, but it must be at least `1`. State an integer in `1:typemax(Int)`, or `nothing` for the plain distance."))
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

Compute the distance-of-distances matrix from a covariance estimator and a data matrix.

# Algorithm

 1. Build a base [`Distance`](@ref) with `de.power` and `de.alg`. Call [`distance`](@ref) on it with `ce`, `X`, `dims` and `kwargs`. The result is the base distance matrix `D`.
 2. Apply `de.metric` to `D` with `Distances.pairwise`, and give it the fields `de.args` and `de.kwargs`. The result is the distance-of-distances matrix.

When `alg` is a [`CanonicalDistance`](@ref), step 1 selects the algorithm of `D` by the type of `ce`. The metric runs after that selection. [`distance`](@ref) holds the redirect table.

`Distances.pairwise` reads each column of `D` as one observation. The call passes `dims = 2` before the splat of `de.kwargs`, so a `dims` in `de.kwargs` replaces it. The axis does not change the result, because a base distance matrix is symmetric. The axis matters only when `de.args` holds a second matrix of a different shape.

# Arguments

  - `de`: The distance-of-distances estimator.
  - `ce`: The covariance estimator.
  - `X`: The data matrix, with one row per observation and one column per asset when `dims = 1`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for the base distance of step 1. They do not reach `Distances.pairwise`, which reads the field `de.kwargs`.

# Returns

  - `D::Matrix{<:Number}`: The matrix of the pairwise distances of distances.

# Related

  - [`DistanceDistance`](@ref)
  - [`Distance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::MatNum;
                  dims::Int = 1, kwargs...)
    D = distance(Distance(; power = de.power, alg = de.alg), ce, X; dims = dims, kwargs...)
    return Distances.pairwise(de.metric, D, de.args...; dims = 2, de.kwargs...)
end
"""
    distance(de::DistanceDistance, rho::MatNum, args...; kwargs...)

Compute the distance-of-distances matrix from a correlation or covariance matrix.

# Algorithm

 1. Build a base [`Distance`](@ref) with `de.power` and `de.alg`. Call [`distance`](@ref) on it with `rho`, `args` and `kwargs`. The result is the base distance matrix `D`.
 2. Apply `de.metric` to `D` with `Distances.pairwise`, and give it the fields `de.args` and `de.kwargs`. The result is the distance-of-distances matrix.

When `alg` is a [`CanonicalDistance`](@ref), the base distance replaces it with [`SimpleDistance`](@ref) before step 1. This route has no covariance estimator, so the redirect table cannot select a row, and the base distance takes the fallback of the table.

`Distances.pairwise` reads the columns of `D`, as it does on the route of the covariance estimator above. The call passes `dims = 2`, and a `dims` in `de.kwargs` replaces it.

# Arguments

  - `de`: The distance-of-distances estimator.
  - `rho`: The correlation or covariance matrix.
  - `args...`: More positional arguments. The method gives them to the base distance of step 1, which ignores them. They are not the field `de.args`, which goes to `Distances.pairwise`.
  - `kwargs...`: More keyword arguments for the base distance of step 1. They do not reach `Distances.pairwise`, which reads the field `de.kwargs`.

# Returns

  - `D::Matrix{<:Number}`: The matrix of the pairwise distances of distances.

# Related

  - [`DistanceDistance`](@ref)
  - [`Distance`](@ref)
  - [`distance`](@ref)
"""
function distance(de::DistanceDistance, rho::MatNum, args...; kwargs...)
    D = distance(Distance(; power = de.power, alg = de.alg), rho, args...; kwargs...)
    return Distances.pairwise(de.metric, D, de.args...; dims = 2, de.kwargs...)
end
"""
    cor_and_dist(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::MatNum;
                 dims::Int = 1, kwargs...)

Compute the correlation matrix and the distance-of-distances matrix from a covariance estimator and a data matrix.

# Algorithm

 1. Build a base [`Distance`](@ref) with `de.power` and `de.alg`. Call [`cor_and_dist`](@ref) on it with `ce`, `X`, `dims` and `kwargs`. One pass gives the correlation matrix `rho` and the base distance matrix `D`.
 2. Apply `de.metric` to `D` with `Distances.pairwise`, and give it the fields `de.args` and `de.kwargs`. The result is the distance-of-distances matrix.
 3. Return `rho` without a change, together with that matrix. The metric reads the distance matrix alone, so the method returns the correlation of the base estimator.

The second element is the matrix that [`distance`](@ref) returns for the same `de`, `ce` and `X`. A caller that needs the two matrices then computes one correlation, not two.

# Arguments

  - `de`: The distance-of-distances estimator.
  - `ce`: The covariance estimator.
  - `X`: The data matrix, with one row per observation and one column per asset when `dims = 1`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for the base distance of step 1. They do not reach `Distances.pairwise`, which reads the field `de.kwargs`.

# Returns

  - `(rho::Matrix{<:Number}, D::Matrix{<:Number})`: The tuple of the correlation matrix and the distance-of-distances matrix.

# Related

  - [`DistanceDistance`](@ref)
  - [`Distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function cor_and_dist(de::DistanceDistance, ce::StatsBase.CovarianceEstimator, X::MatNum;
                      dims::Int = 1, kwargs...)
    rho, D = cor_and_dist(Distance(; power = de.power, alg = de.alg), ce, X; dims = dims,
                          kwargs...)
    return rho, Distances.pairwise(de.metric, D, de.args...; dims = 2, de.kwargs...)
end

export DistanceDistance
