"""
    struct DistanceCovariance{T1, T2, T3, T4, T5} <: AbstractCovarianceEstimator
        dist::T1
        args::T2
        kwargs::T3
        w::T4
        ex::T5
    end

A flexible container type for configuring and applying distance-based covariance estimators in PortfolioOptimisers.jl.

`DistanceCovariance` encapsulates all components required for distance covariance or correlation estimation, including the distance metric, additional arguments and keyword arguments for the metric, optional weights, and parallel execution strategy. This enables modular and extensible workflows for robust covariance estimation using distance statistics.

# Fields

  - `dist`: Distance metric used for pairwise computations.
  - `args`: Additional positional arguments for the distance metric.
  - `kwargs`: Additional keyword arguments for the distance metric.
  - `w`: Optional weights for observations.
  - `ex`: Parallel execution strategy.

# Constructor

    DistanceCovariance(; dist::Distances.Metric = Distances.Euclidean(), args::Tuple = (),
                       kwargs::NamedTuple = (;), w::Option{<:AbstractWeights} = nothing,
                       ex::FLoops.Transducers.Executor = ThreadedEx())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> DistanceCovariance()
DistanceCovariance
    dist ┼ Distances.Euclidean: Distances.Euclidean(0.0)
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
       w ┼ nothing
      ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`Distances.Metric`](https://github.com/JuliaStats/Distances.jl)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-ex)
"""
struct DistanceCovariance{T1, T2, T3, T4, T5} <: AbstractCovarianceEstimator
    dist::T1
    args::T2
    kwargs::T3
    w::T4
    ex::T5
    function DistanceCovariance(dist::Distances.Metric, args::Tuple, kwargs::NamedTuple,
                                w::Option{<:AbstractWeights},
                                ex::FLoops.Transducers.Executor)
        return new{typeof(dist), typeof(args), typeof(kwargs), typeof(w), typeof(ex)}(dist,
                                                                                      args,
                                                                                      kwargs,
                                                                                      w, ex)
    end
end
function DistanceCovariance(; dist::Distances.Metric = Distances.Euclidean(),
                            args::Tuple = (), kwargs::NamedTuple = (;),
                            w::Option{<:AbstractWeights} = nothing,
                            ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())
    return DistanceCovariance(dist, args, kwargs, w, ex)
end
function factory(ce::DistanceCovariance, w::Option{<:AbstractWeights} = nothing)
    return DistanceCovariance(; dist = ce.dist, args = ce.args, kwargs = ce.kwargs,
                              w = isnothing(w) ? ce.w : w, ex = ce.ex)
end
"""
    cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)

Compute the distance correlation between two vectors using a configured [`DistanceCovariance`](@ref) estimator.

This function computes the distance correlation between `v1` and `v2` using the specified distance metric, optional weights, and any additional arguments or keyword arguments provided in the estimator. The computation follows the standard distance correlation procedure, centering the pairwise distance matrices and normalizing the result.

# Arguments

  - `ce`: Distance covariance estimator.
  - `v1`: First data vector.
  - `v2`: Second data vector.

# Returns

  - `rho::Float64`: The computed distance correlation between `v1` and `v2`.

# Details

 1. Computes pairwise distance matrices for `v1` and `v2` using the estimator's metric and configuration.
 2. Centers the distance matrices by subtracting row and column means and adding the grand mean.
 3. Computes the squared distance covariance and normalizes to obtain the distance correlation.

# Validation

  - `length(v1) == length(v2)`.
  - `length(v1) > 1`.

# Related

  - [`DistanceCovariance`](@ref)
  - [`cor_distance(ce::DistanceCovariance, X::MatNum)`](@ref)
"""
function cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)
    N = length(v1)
    @argcheck(1 < N, DimensionMismatch("1 < length(v1) must hold. Got\nlength(v1) => $N"))
    @argcheck(N == length(v2), DimensionMismatch)
    N2 = N^2
    a, b = if isnothing(ce.w)
        Distances.pairwise(ce.dist, v1, ce.args...; ce.kwargs...),
        Distances.pairwise(ce.dist, v2, ce.args...; ce.kwargs...)
    else
        Distances.pairwise(ce.dist, v1 ⊙ ce.w, ce.args...; ce.kwargs...),
        Distances.pairwise(ce.dist, v2 ⊙ ce.w, ce.args...; ce.kwargs...)
    end
    mu_a1, mu_b1 = mean(a; dims = 1), mean(b; dims = 1)
    mu_a2, mu_b2 = mean(a; dims = 2), mean(b; dims = 2)
    mu_a3, mu_b3 = mean(a), mean(b)
    A = a .- mu_a1 .- mu_a2 .+ mu_a3
    B = b .- mu_b1 .- mu_b2 .+ mu_b3
    dcov2_xx = dot(A, A) / N2
    dcov2_xy = dot(A, B) / N2
    dcov2_yy = dot(B, B) / N2
    return sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))
end
"""
    cor_distance(ce::DistanceCovariance, X::MatNum)

Compute the pairwise distance correlation matrix for all columns in a data matrix using a configured [`DistanceCovariance`](@ref) estimator.

This function computes the distance correlation between each pair of columns in `X`, using the specified distance metric, optional weights, and parallel execution strategy. The resulting matrix is symmetric, with each entry representing the distance correlation between two assets.

# Arguments

  - `ce`: Distance covariance estimator.
  - `X`: Data matrix (observations × assets).

# Returns

  - `rho::Matrix{<:Number}`: Distance correlation matrix.

# Details

  - Iterates over all pairs of columns in `X`, computing the distance correlation for each pair using [`cor_distance(ce, v1, v2)`](@ref).
  - Parallelizes computation using the estimator's `ex` field.

# Related

  - [`DistanceCovariance`](@ref)
  - [`cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)`](@ref)
"""
function cor_distance(ce::DistanceCovariance, X::MatNum)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    FLoops.@floop ce.ex for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            rho[j, i] = rho[i, j] = cor_distance(ce, view(X, :, i), xj)
        end
    end
    return rho
end
"""
    Statistics.cor(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the pairwise distance correlation matrix for all columns in a data matrix using a configured [`DistanceCovariance`](@ref) estimator.

# Arguments

  - `ce`: Distance covariance estimator.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute correlations.
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of pairwise distance correlations.

# Validation

  - `dims` is either `1` or `2`.

# Examples

```jldoctest
julia> ce = DistanceCovariance()
DistanceCovariance
    dist ┼ Distances.Euclidean: Distances.Euclidean(0.0)
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
       w ┼ nothing
      ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()

julia> X = [1.0 2.0; 2.0 4.0; 3.0 6.0];

julia> cor(ce, X)
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```

# Related

  - [`DistanceCovariance`](@ref)
  - [`cor_distance(ce::DistanceCovariance, X::MatNum)`](@ref)
  - [`cov(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return cor_distance(ce, X)
end
"""
    cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)

Compute the distance covariance between two vectors using a configured [`DistanceCovariance`](@ref) estimator.

This function computes the distance covariance between `v1` and `v2` using the specified distance metric, optional weights, and any additional arguments or keyword arguments provided in the estimator. The computation follows the standard distance covariance procedure, centering the pairwise distance matrices and aggregating the result.

# Arguments

  - `ce`: Distance covariance estimator.
  - `v1`: First data vector.
  - `v2`: Second data vector.

# Returns

  - `rho::Number`: The computed distance covariance between `v1` and `v2`.

# Details

 1. Computes pairwise distance matrices for `v1` and `v2` using the estimator's metric and configuration.
 2. Centers the distance matrices by subtracting row and column means and adding the grand mean.
 3. Computes the squared distance covariance and returns its square root.

# Validation

  - `length(v1) == length(v2)`.
  - `length(v1) > 1`.

# Related

  - [`DistanceCovariance`](@ref)
  - [`cov_distance(ce::DistanceCovariance, X::MatNum)`](@ref)
"""
function cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)
    N = length(v1)
    @argcheck(1 < N, DimensionMismatch("1 < length(v1) must hold. Got\nlength(v1) => $N"))
    @argcheck(N == length(v2), DimensionMismatch)
    N2 = N^2
    a, b = if isnothing(ce.w)
        Distances.pairwise(ce.dist, v1, ce.args...; ce.kwargs...),
        Distances.pairwise(ce.dist, v2, ce.args...; ce.kwargs...)
    else
        Distances.pairwise(ce.dist, v1 ⊙ ce.w, ce.args...; ce.kwargs...),
        Distances.pairwise(ce.dist, v2 ⊙ ce.w, ce.args...; ce.kwargs...)
    end
    mu_a1, mu_b1 = mean(a; dims = 1), mean(b; dims = 1)
    mu_a2, mu_b2 = mean(a; dims = 2), mean(b; dims = 2)
    mu_a3, mu_b3 = mean(a), mean(b)
    A = a .- mu_a1 .- mu_a2 .+ mu_a3
    B = b .- mu_b1 .- mu_b2 .+ mu_b3
    dcov2_xy = dot(A, B) / N2
    return sqrt(dcov2_xy)
end
"""
    cov_distance(ce::DistanceCovariance, X::MatNum)

Compute the pairwise distance covariance matrix for all columns in a data matrix using a configured [`DistanceCovariance`](@ref) estimator.

This function computes the distance covariance between each pair of columns in `X`, using the specified distance metric, optional weights, and parallel execution strategy. The resulting matrix is symmetric, with each entry representing the distance covariance between two assets.

# Arguments

  - `ce`: Distance covariance estimator.
  - `X`: Data matrix (observations × assets).

# Returns

  - `sigma::Matrix{<:Number}`: Symmetric matrix of pairwise distance covariances.

# Details

  - Iterates over all pairs of columns in `X`, computing the distance covariance for each pair using [`cov_distance(ce, v1, v2)`](@ref).
  - Parallelizes computation using the estimator's `ex` field.

# Related

  - [`DistanceCovariance`](@ref)
  - [`cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)`](@ref)
"""
function cov_distance(ce::DistanceCovariance, X::MatNum)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    FLoops.@floop ce.ex for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            rho[j, i] = rho[i, j] = cov_distance(ce, view(X, :, i), xj)
        end
    end
    return rho
end
"""
    Statistics.cov(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the pairwise distance covariance matrix for all columns in a data matrix using a configured [`DistanceCovariance`](@ref) estimator.

# Arguments

  - `ce`: Distance covariance estimator.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute covariances.
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `sigma::Matrix{<:Number}`: Symmetric matrix of pairwise distance covariances.

# Validation

  - `dims` is either `1` or `2`.

# Examples

```jldoctest
julia> ce = DistanceCovariance()
DistanceCovariance
    dist ┼ Distances.Euclidean: Distances.Euclidean(0.0)
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
       w ┼ nothing
      ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()

julia> X = [1.0 2.0; 2.0 4.0; 3.0 6.0];

julia> cov(ce, X)
2×2 Matrix{Float64}:
 0.702728  0.993808
 0.993808  1.40546
```

# Related

  - [`DistanceCovariance`](@ref)
  - [`cov_distance(ce::DistanceCovariance, X::MatNum)`](@ref)
  - [`cor(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return cov_distance(ce, X)
end

export DistanceCovariance
