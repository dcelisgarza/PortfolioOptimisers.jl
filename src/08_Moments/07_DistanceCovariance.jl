"""
$(DocStringExtensions.TYPEDEF)

A flexible container type for configuring and applying distance-based covariance estimators in `PortfolioOptimisers.jl`.

`DistanceCovariance` encapsulates all components required for distance covariance or correlation estimation, including the distance metric, additional arguments and keyword arguments for the metric, optional weights, and parallel execution strategy.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistanceCovariance(;
        metric::Distances.Metric = Distances.Euclidean(),
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        w::Option{<:ObsWeights} = nothing,
        ex::FLoops.Transducers.Executor = ThreadedEx()
    ) -> DistanceCovariance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> DistanceCovariance()
DistanceCovariance
  metric ┼ Distances.Euclidean: Distances.Euclidean(0.0)
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
@concrete struct DistanceCovariance <: AbstractCovarianceEstimator
    "$(arg_dict[:metric])"
    metric
    "$(arg_dict[:metric_args])"
    args
    "$(arg_dict[:metric_kwargs])"
    kwargs
    "$(arg_dict[:oow])"
    w
    "$(arg_dict[:ex])"
    ex
    function DistanceCovariance(metric::Distances.Metric, args::Tuple, kwargs::NamedTuple,
                                w::Option{<:ObsWeights}, ex::FLoops.Transducers.Executor)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(metric), typeof(args), typeof(kwargs), typeof(w), typeof(ex)}(metric,
                                                                                        args,
                                                                                        kwargs,
                                                                                        w,
                                                                                        ex)
    end
end
function DistanceCovariance(; metric::Distances.Metric = Distances.Euclidean(),
                            args::Tuple = (), kwargs::NamedTuple = (;),
                            w::Option{<:ObsWeights} = nothing,
                            ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())::DistanceCovariance
    return DistanceCovariance(metric, args, kwargs, w, ex)
end
"""
    factory(ce::DistanceCovariance, w::ObsWeights) -> DistanceCovariance

Return a new [`DistanceCovariance`](@ref) estimator with observation weights `w`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Examples

```jldoctest
julia> ce = DistanceCovariance();

julia> factory(ce, StatsBase.Weights([0.2, 0.3, 0.5]))
DistanceCovariance
  metric ┼ Distances.Euclidean: Distances.Euclidean(0.0)
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
       w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
      ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`DistanceCovariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::DistanceCovariance, w::ObsWeights)::DistanceCovariance
    return DistanceCovariance(; metric = ce.metric, args = ce.args, kwargs = ce.kwargs,
                              w = w, ex = ce.ex)
end
"""
    calc_pairwise_dists(ce::DistanceCovariance, v1::VecNum, v2::VecNum, w::Option{<:StatsBase.AbstractWeights}) -> (MatNum, MatNum)

Compute pairwise distance matrices between two vectors using the configured metric.

Internal helper used in distance correlation computation. Handles weighted and unweighted cases.

# Arguments

  - `ce`: [`DistanceCovariance`](@ref) estimator with metric configuration.
  - `v1`, `v2`: Data vectors.
  - `w`: Observation weights (`nothing` for unweighted, `StatsBase.AbstractWeights` for weighted).

# Returns

  - Tuple of pairwise distance matrices `(D1, D2)`.

# Related

  - [`DistanceCovariance`](@ref)
  - [`cor_distance`](@ref)
"""
function calc_pairwise_dists(ce::DistanceCovariance, v1::VecNum, v2::VecNum, ::Nothing)
    return Distances.pairwise(ce.metric, v1, ce.args...; ce.kwargs...),
           Distances.pairwise(ce.metric, v2, ce.args...; ce.kwargs...)
end
function calc_pairwise_dists(ce::DistanceCovariance, v1::VecNum, v2::VecNum,
                             w::StatsBase.AbstractWeights)
    return Distances.pairwise(ce.metric, v1 ⊙ w, ce.args...; ce.kwargs...),
           Distances.pairwise(ce.metric, v2 ⊙ w, ce.args...; ce.kwargs...)
end
"""
    cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)

Compute the distance correlation between two vectors using a configured [`DistanceCovariance`](@ref) estimator.

This function computes the distance correlation between `v1` and `v2` using the specified distance metric, optional weights, and any additional arguments or keyword arguments provided in the estimator. The computation follows the standard distance correlation procedure, centering the pairwise distance matrices and normalizing the result.

# Mathematical definition

Let ``a_{kl} = d(v_{1k}, v_{1l})`` and ``b_{kl} = d(v_{2k}, v_{2l})`` be pairwise distance matrices. Define doubly-centered versions:

```math
\\begin{align}
A_{kl} &= a_{kl} - \\bar{a}_{k\\cdot} - \\bar{a}_{\\cdot l} + \\bar{a}_{\\cdot\\cdot}\\,, \\\\
B_{kl} &= b_{kl} - \\bar{b}_{k\\cdot} - \\bar{b}_{\\cdot l} + \\bar{b}_{\\cdot\\cdot}\\,.
\\end{align}
```

Where:

  - ``a_{kl}``, ``b_{kl}``: Pairwise distances between observations ``k`` and ``l``.
  - ``\\bar{a}_{k\\cdot}``: ``k``-th row mean of ``\\mathbf{a}``.
  - ``\\bar{a}_{\\cdot l}``: ``l``-th column mean of ``\\mathbf{a}``.
  - ``\\bar{a}_{\\cdot\\cdot}``: Grand mean of ``\\mathbf{a}``.

The squared distance covariances and distance correlation are:

```math
\\begin{align}
\\widehat{\\mathrm{dCov}}^2(X,X) &= \\frac{\\mathbf{A}:\\mathbf{A}}{n^2}\\,, \\\\
\\widehat{\\mathrm{dCov}}^2(X,Y) &= \\frac{\\mathbf{A}:\\mathbf{B}}{n^2}\\,, \\\\
\\widehat{\\mathrm{dCov}}^2(Y,Y) &= \\frac{\\mathbf{B}:\\mathbf{B}}{n^2}\\,.
\\end{align}
```

Where:

  - ``n``: Number of observations.
  - ``\\mathbf{A}:\\mathbf{B} = \\sum_{k,l} A_{kl} B_{kl}``: Frobenius inner product.

```math
\\begin{align}
\\hat{R}_{\\mathrm{dist}}(X, Y) &= \\frac{\\sqrt{\\widehat{\\mathrm{dCov}}^2(X,Y)}}{\\sqrt{\\sqrt{\\widehat{\\mathrm{dCov}}^2(X,X)} \\cdot \\sqrt{\\widehat{\\mathrm{dCov}}^2(Y,Y)}}}\\,.
\\end{align}
```

Where:

  - ``\\hat{R}_{\\mathrm{dist}}(X, Y)``: Distance correlation between ``X`` and ``Y``.

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
function cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum,
                      w::Option{<:StatsBase.AbstractWeights} = nothing)
    N = length(v1)
    @argcheck(1 < N, DimensionMismatch("1 < length(v1) must hold. Got\nlength(v1) => $N"))
    @argcheck(N == length(v2), DimensionMismatch)
    N2 = N^2
    a, b = calc_pairwise_dists(ce, v1, v2, w)
    mu_a1, mu_b1 = Statistics.mean(a; dims = 1), Statistics.mean(b; dims = 1)
    mu_a2, mu_b2 = Statistics.mean(a; dims = 2), Statistics.mean(b; dims = 2)
    mu_a3, mu_b3 = Statistics.mean(a), Statistics.mean(b)
    A = a .- mu_a1 .- mu_a2 .+ mu_a3
    B = b .- mu_b1 .- mu_b2 .+ mu_b3
    dcov2_xx = LinearAlgebra.dot(A, A) / N2
    dcov2_xy = LinearAlgebra.dot(A, B) / N2
    dcov2_yy = LinearAlgebra.dot(B, B) / N2
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
function cor_distance(ce::DistanceCovariance, X::MatNum,
                      w::Option{<:StatsBase.AbstractWeights} = nothing)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    FLoops.@floop ce.ex for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            rho[j, i] = rho[i, j] = cor_distance(ce, view(X, :, i), xj, w)
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
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of pairwise distance correlations.

# Validation

  - `dims` is either `1` or `2`.

# Examples

```jldoctest
julia> ce = DistanceCovariance()
DistanceCovariance
  metric ┼ Distances.Euclidean: Distances.Euclidean(0.0)
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
    w = get_observation_weights(ce.w, X)
    return cor_distance(ce, X, w)
end
"""
    cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)

Compute the distance covariance between two vectors using a configured [`DistanceCovariance`](@ref) estimator.

This function computes the distance covariance between `v1` and `v2` using the specified distance metric, optional weights, and any additional arguments or keyword arguments provided in the estimator. The computation follows the standard distance covariance procedure, centering the pairwise distance matrices and aggregating the result.

# Mathematical definition

Using the same doubly-centered matrices ``\\mathbf{A}`` and ``\\mathbf{B}`` as in [`cor_distance`](@ref):

```math
\\begin{align}
\\widehat{\\mathrm{dCov}}(X, Y) &= \\sqrt{\\left|\\frac{\\mathbf{A}:\\mathbf{B}}{n^2}\\right|}\\,.
\\end{align}
```

Where:

  - ``\\widehat{\\mathrm{dCov}}(X, Y)``: Distance covariance between ``X`` and ``Y``.
  - ``n``: Number of observations.
  - ``\\mathbf{A}:\\mathbf{B} = \\sum_{k,l} A_{kl} B_{kl}``: Frobenius inner product of doubly-centered distance matrices.

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
function cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum,
                      w::Option{<:StatsBase.AbstractWeights} = nothing)
    N = length(v1)
    @argcheck(1 < N, DimensionMismatch("1 < length(v1) must hold. Got\nlength(v1) => $N"))
    @argcheck(N == length(v2), DimensionMismatch)
    N2 = N^2
    a, b = calc_pairwise_dists(ce, v1, v2, w)
    mu_a1, mu_b1 = Statistics.mean(a; dims = 1), Statistics.mean(b; dims = 1)
    mu_a2, mu_b2 = Statistics.mean(a; dims = 2), Statistics.mean(b; dims = 2)
    mu_a3, mu_b3 = Statistics.mean(a), Statistics.mean(b)
    A = a .- mu_a1 .- mu_a2 .+ mu_a3
    B = b .- mu_b1 .- mu_b2 .+ mu_b3
    dcov2_xy = LinearAlgebra.dot(A, B) / N2
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
function cov_distance(ce::DistanceCovariance, X::MatNum,
                      w::Option{<:StatsBase.AbstractWeights} = nothing)
    N = size(X, 2)
    rho = Matrix{eltype(X)}(undef, N, N)
    FLoops.@floop ce.ex for j in axes(X, 2)
        xj = view(X, :, j)
        for i in 1:j
            rho[j, i] = rho[i, j] = cov_distance(ce, view(X, :, i), xj, w)
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
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Returns

  - `sigma::Matrix{<:Number}`: Symmetric matrix of pairwise distance covariances.

# Validation

  - `dims` is either `1` or `2`.

# Examples

```jldoctest
julia> ce = DistanceCovariance()
DistanceCovariance
  metric ┼ Distances.Euclidean: Distances.Euclidean(0.0)
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
    w = get_observation_weights(ce.w, X)
    return cov_distance(ce, X, w)
end

export DistanceCovariance
