"""
$(DocStringExtensions.TYPEDEF)

Measures linear and non-linear codependence from doubly-centred pairwise distance matrices.

The statistic is the distance covariance, which is zero if and only if the two series are independent. `metric`, `args` and `kwargs` configure the pairwise distance; `w` weights the observations and `ex` selects the parallel execution strategy.

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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@wprop`-tagged field is automatically propagated:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).

## Observation weight parameters

When [`obs_weights_view`](@ref) is called on this type, the following fields are automatically indexed to the selected observations:

  - `w`: Indexed to the selected observations via [`obs_weights_view`](@ref).

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
  - [`factory`](@ref)
  - [`obs_weights_view`](@ref)
  - [`cor_distance`](@ref)
  - [`cov_distance`](@ref)

# References

  - $(ref_dict[:szekely2007])
  - $(ref_dict[:cajas2025]) Section 6.1.5, equations 6.9 to 6.13.
"""
@propagatable @concrete struct DistanceCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:metric])
    """
    metric
    """
    $(field_dict[:metric_args])
    """
    args
    """
    $(field_dict[:metric_kwargs])
    """
    kwargs
    """
    $(field_dict[:oow])
    """
    @wprop w
    """
    $(field_dict[:ex])
    """
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
    calc_pairwise_dists(ce::DistanceCovariance, v1::VecNum, v2::VecNum) -> (MatNum, MatNum)

Compute pairwise distance matrices between two vectors using the configured metric.

Internal helper used in distance correlation and distance covariance computation. The observation weights never reach it. They weight the statistic that [`calc_centred_dists`](@ref) and [`calc_dcov2`](@ref) build, and not the data that the metric measures.

# Algorithm

 1. Apply the estimator's `metric` to `v1` and to `v2` as the caller gave them, and pass `ce.args` and `ce.kwargs` to `Distances.pairwise`.
 2. Return the two matrices in the order `(D1, D2)`. Each is ``T \\times T`` for a pair of series of ``T`` observations, and `D1[t, s]` is the distance between observations ``t`` and ``s`` of the same series.

# Arguments

  - `ce`: [`DistanceCovariance`](@ref) estimator with metric configuration.
  - `v1`, `v2`: Data vectors.

# Returns

  - Tuple of pairwise distance matrices `(D1, D2)`.

# Related

  - [`DistanceCovariance`](@ref)
  - [`calc_centred_dists`](@ref)
  - [`calc_dcov2`](@ref)
  - [`cor_distance`](@ref)
"""
function calc_pairwise_dists(ce::DistanceCovariance, v1::VecNum, v2::VecNum)
    return Distances.pairwise(ce.metric, v1, ce.args...; ce.kwargs...),
           Distances.pairwise(ce.metric, v2, ce.args...; ce.kwargs...)
end
"""
    calc_centred_dists(a::MatNum, w::Option{<:StatsBase.AbstractWeights}) -> MatNum

Doubly centre a pairwise distance matrix, weighting every observation by `w`.

Internal helper used in distance correlation and distance covariance computation. The weights enter the three means, so a constant weight vector cancels from each of them and returns the unweighted matrix.

# Algorithm

 1. Without weights, take the ordinary row means, column means and grand mean of `a`.
 2. With weights, take the same three means weighted by `w`. Each is normalised by the sum of the weights that it uses, so the scale of `w` cancels.
 3. Subtract the row means and the column means from `a`, then add the grand mean back, giving `A`.

# Arguments

  - `a`: Pairwise distance matrix.
  - `w`: Observation weights, or `nothing` for the unweighted statistic.

# Returns

  - `A::MatNum`: Doubly centred distance matrix.

# Related

  - [`DistanceCovariance`](@ref)
  - [`calc_pairwise_dists`](@ref)
  - [`calc_dcov2`](@ref)
  - [`cor_distance`](@ref)
"""
function calc_centred_dists(a::MatNum, ::Nothing)
    return a .- Statistics.mean(a; dims = 1) .- Statistics.mean(a; dims = 2) .+
           Statistics.mean(a)
end
function calc_centred_dists(a::MatNum, w::StatsBase.AbstractWeights)
    mu2 = Statistics.mean(a, w; dims = 2)
    return a .- Statistics.mean(a, w; dims = 1) .- mu2 .+ Statistics.mean(vec(mu2), w)
end
"""
    calc_dcov2(A::MatNum, B::MatNum, w::Option{<:StatsBase.AbstractWeights}) -> Number

Contract two doubly centred distance matrices into a squared distance covariance.

Internal helper used in distance correlation and distance covariance computation. The weights enter both indices of the contraction, so a constant weight vector cancels and returns the unweighted value.

# Algorithm

 1. Without weights, take the Frobenius inner product of `A` and `B`, and divide it by the number of entries of `A`.
 2. With weights, take the element-wise product of `A` and `B`, contract it with `w` on both of its indices, and divide by the square of the sum of the weights.

# Arguments

  - `A`, `B`: Doubly centred distance matrices, as [`calc_centred_dists`](@ref) returns them.
  - `w`: Observation weights, or `nothing` for the unweighted statistic.

# Returns

  - `dcov2::Number`: Squared distance covariance of the two matrices.

# Related

  - [`DistanceCovariance`](@ref)
  - [`calc_pairwise_dists`](@ref)
  - [`calc_centred_dists`](@ref)
  - [`cov_distance`](@ref)
"""
function calc_dcov2(A::MatNum, B::MatNum, ::Nothing)
    return LinearAlgebra.dot(A, B) / length(A)
end
function calc_dcov2(A::MatNum, B::MatNum, w::StatsBase.AbstractWeights)
    return LinearAlgebra.dot(w, A ⊙ B, w) / sum(w)^2
end
"""
    cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)

Compute the distance correlation between two vectors using a configured [`DistanceCovariance`](@ref) estimator.

This function computes the distance correlation between `v1` and `v2` using the specified distance metric, optional weights, and any additional arguments or keyword arguments provided in the estimator. The computation follows the standard distance correlation procedure, centering the pairwise distance matrices and normalizing the result.

# Mathematical definition

Let ``a_{ts} = d(v_{1t}, v_{1s})`` and ``b_{ts} = d(v_{2t}, v_{2s})`` be pairwise distance matrices. Define doubly-centered versions, whose three means carry the observation weights:

```math
\\begin{align}
\\bar{a}_{t\\cdot} &= \\frac{\\sum\\limits_{s=1}^{T} w_{s} a_{ts}}{\\sum\\limits_{s=1}^{T} w_{s}}\\,, \\\\
\\bar{a}_{\\cdot s} &= \\frac{\\sum\\limits_{t=1}^{T} w_{t} a_{ts}}{\\sum\\limits_{t=1}^{T} w_{t}}\\,, \\\\
\\bar{a}_{\\cdot\\cdot} &= \\frac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{s=1}^{T} w_{t} w_{s} a_{ts}}{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}\\,, \\\\
A_{ts} &= a_{ts} - \\bar{a}_{t\\cdot} - \\bar{a}_{\\cdot s} + \\bar{a}_{\\cdot\\cdot}\\,, \\\\
B_{ts} &= b_{ts} - \\bar{b}_{t\\cdot} - \\bar{b}_{\\cdot s} + \\bar{b}_{\\cdot\\cdot}\\,.
\\end{align}
```

Where:

  - ``a_{ts}``, ``b_{ts}``: Pairwise distances between observations ``t`` and ``s``.
  - ``\\bar{a}_{t\\cdot}``: ``t``-th weighted row mean of ``\\mathbf{a}``.
  - ``\\bar{a}_{\\cdot s}``: ``s``-th weighted column mean of ``\\mathbf{a}``.
  - ``\\bar{a}_{\\cdot\\cdot}``: Weighted grand mean of ``\\mathbf{a}``.
  - ``A_{ts}``, ``B_{ts}``: Doubly centred pairwise distances.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])

The three means of ``\\mathbf{b}`` take the form that the three means of ``\\mathbf{a}`` take.

The squared distance covariances and distance correlation are:

```math
\\begin{align}
\\widehat{\\mathrm{dCov}}^2(X,X) &= \\frac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{s=1}^{T} w_{t} w_{s} A_{ts} A_{ts}}{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}\\,, \\\\
\\widehat{\\mathrm{dCov}}^2(X,Y) &= \\frac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{s=1}^{T} w_{t} w_{s} A_{ts} B_{ts}}{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}\\,, \\\\
\\widehat{\\mathrm{dCov}}^2(Y,Y) &= \\frac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{s=1}^{T} w_{t} w_{s} B_{ts} B_{ts}}{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}\\,.
\\end{align}
```

Where:

  - ``\\widehat{\\mathrm{dCov}}^2(X,Y)``: Squared distance covariance of ``X`` and ``Y``.

```math
\\begin{align}
\\hat{R}_{\\mathrm{dist}}(X, Y) &= \\frac{\\sqrt{\\widehat{\\mathrm{dCov}}^2(X,Y)}}{\\sqrt{\\sqrt{\\widehat{\\mathrm{dCov}}^2(X,X)} \\cdot \\sqrt{\\widehat{\\mathrm{dCov}}^2(Y,Y)}}}\\,.
\\end{align}
```

Where:

  - ``\\hat{R}_{\\mathrm{dist}}(X, Y)``: Distance correlation between ``X`` and ``Y``.

Each quotient above divides two forms of one degree in the weights, so a constant weight cancels from all of them. The value ``w_{t} = 1`` reduces the three squared distance covariances to ``\\mathbf{A}:\\mathbf{B} / T^{2}``, the Frobenius inner product ``\\sum_{t,\\,s} A_{ts} B_{ts}`` over ``T^{2}``, which is the unweighted statistic.

# Algorithm

 1. Build the two pairwise distance matrices ``\\mathbf{a}`` and ``\\mathbf{b}`` with [`calc_pairwise_dists`](@ref), which carries the estimator's metric, its `args` and its `kwargs`.
 2. Doubly centre each matrix with [`calc_centred_dists`](@ref), which weights the three means it subtracts and adds, giving ``\\mathbf{A}`` and ``\\mathbf{B}``.
 3. Contract the three pairs ``(\\mathbf{A}, \\mathbf{A})``, ``(\\mathbf{A}, \\mathbf{B})`` and ``(\\mathbf{B}, \\mathbf{B})`` with [`calc_dcov2`](@ref), giving the three squared distance covariances.
 4. Return ``\\sqrt{\\widehat{\\mathrm{dCov}}^2(X,Y)}`` divided by the square root of the product of the two remaining square roots.

# Arguments

  - `ce`: Distance covariance estimator.
  - `v1`: First data vector.
  - `v2`: Second data vector.
  - `w`: Observation weights, or `nothing` for the unweighted statistic.

# Validation

  - `length(v1) == length(v2)`.
  - `length(v1) > 1`.

# Returns

  - `rho::Float64`: The computed distance correlation between `v1` and `v2`. A series against itself gives exactly `1.0`.

# Related

  - [`DistanceCovariance`](@ref)
  - [`calc_pairwise_dists`](@ref)
  - [`calc_centred_dists`](@ref)
  - [`calc_dcov2`](@ref)
  - [`cor_distance(ce::DistanceCovariance, X::MatNum)`](@ref)
"""
function cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum,
                      w::Option{<:StatsBase.AbstractWeights} = nothing)
    N = length(v1)
    @argcheck(1 < N, DimensionMismatch("1 < length(v1) must hold. Got\nlength(v1) => $N"))
    @argcheck(N == length(v2), DimensionMismatch)
    a, b = calc_pairwise_dists(ce, v1, v2)
    A, B = calc_centred_dists(a, w), calc_centred_dists(b, w)
    dcov2_xx = calc_dcov2(A, A, w)
    dcov2_xy = calc_dcov2(A, B, w)
    dcov2_yy = calc_dcov2(B, B, w)
    return sqrt(dcov2_xy) / sqrt(sqrt(dcov2_xx) * sqrt(dcov2_yy))
end
"""
    cor_distance(ce::DistanceCovariance, X::MatNum,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)

Compute the pairwise distance correlation matrix for all columns in a data matrix using a configured [`DistanceCovariance`](@ref) estimator.

This function computes the distance correlation between each pair of columns in `X`, using the specified distance metric, optional weights, and parallel execution strategy. The resulting matrix is symmetric, with each entry representing the distance correlation between two assets.

# Algorithm

 1. Allocate an ``N \\times N`` matrix, where ``N`` is the number of columns of `X`. The result is indexed by asset and never by observation, so a non-square `X` cannot hide a transposed index.
 2. For each column `j`, and for each column `i` at or below `j`, call [`cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)`](@ref) on the two columns and write the value into both `rho[i, j]` and `rho[j, i]`. The estimator's `ex` field runs the outer loop.
 3. Return the symmetric matrix. Its diagonal is exactly `1.0`, because the pair `(j, j)` is one of the pairs step 2 computes.

# Arguments

  - `ce`: Distance covariance estimator.
  - `X`: Data matrix (observations × assets).
  - `w`: Observation weights, or `nothing` for the unweighted statistic.

# Returns

  - `rho::Matrix{<:Number}`: Distance correlation matrix.

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

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), which transposes it when `dims` is `2` and refuses any other value.
 2. Resolve the estimator's `w` field against the oriented matrix with [`get_observation_weights`](@ref), giving `nothing` for an unweighted estimator.
 3. Return [`cor_distance(ce::DistanceCovariance, X::MatNum)`](@ref) of the oriented matrix and those weights.

# Arguments

  - `ce`: Distance covariance estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of pairwise distance correlations, with a diagonal of exactly `1.0`.

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
    X = dims_oriented(dims, X)
    w = get_observation_weights(ce.w, X)
    return cor_distance(ce, X, w)
end
"""
    cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)

Compute the distance covariance between two vectors using a configured [`DistanceCovariance`](@ref) estimator.

This function computes the distance covariance between `v1` and `v2` using the specified distance metric, optional weights, and any additional arguments or keyword arguments provided in the estimator. The computation follows the standard distance covariance procedure, centering the pairwise distance matrices and aggregating the result.

# Mathematical definition

Using the same doubly-centered matrices ``\\mathbf{A}`` and ``\\mathbf{B}`` as in [`cor_distance`](@ref):

```math
\\begin{align}
\\widehat{\\mathrm{dCov}}(X, Y) &= \\sqrt{\\frac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{s=1}^{T} w_{t} w_{s} A_{ts} B_{ts}}{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}}\\,.
\\end{align}
```

Where:

  - ``\\widehat{\\mathrm{dCov}}(X, Y)``: Distance covariance between ``X`` and ``Y``.
  - ``A_{ts}``, ``B_{ts}``: Doubly centred pairwise distances.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])

The value ``w_{t} = 1`` reduces the quotient to ``\\mathbf{A}:\\mathbf{B} / T^{2}``, the Frobenius inner product ``\\sum_{t,\\,s} A_{ts} B_{ts}`` over ``T^{2}``, which is the unweighted statistic.

The square root takes no absolute value. It needs none: the doubly-centred V-statistic is non-negative for a metric of strong negative type, which the Euclidean default is.

# Algorithm

 1. Build the two pairwise distance matrices ``\\mathbf{a}`` and ``\\mathbf{b}`` with [`calc_pairwise_dists`](@ref), which carries the estimator's metric, its `args` and its `kwargs`.
 2. Doubly centre each matrix with [`calc_centred_dists`](@ref), which weights the three means it subtracts and adds, giving ``\\mathbf{A}`` and ``\\mathbf{B}``.
 3. Contract the pair ``(\\mathbf{A}, \\mathbf{B})`` with [`calc_dcov2`](@ref), giving the squared distance covariance.
 4. Return the square root of that value. Steps 1 and 2 are those of [`cor_distance`](@ref); this method takes one of the three contractions and takes no ratio.

# Arguments

  - `ce`: Distance covariance estimator.
  - `v1`: First data vector.
  - `v2`: Second data vector.
  - `w`: Observation weights, or `nothing` for the unweighted statistic.

# Validation

  - `length(v1) == length(v2)`.
  - `length(v1) > 1`.

# Returns

  - `rho::Number`: The computed distance covariance between `v1` and `v2`. A series against itself gives the distance standard deviation ``\\widehat{\\mathrm{dVar}}^{1/2}(X)``, which is not the sample standard deviation.

# Related

  - [`DistanceCovariance`](@ref)
  - [`calc_pairwise_dists`](@ref)
  - [`calc_centred_dists`](@ref)
  - [`calc_dcov2`](@ref)
  - [`cov_distance(ce::DistanceCovariance, X::MatNum)`](@ref)
"""
function cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum,
                      w::Option{<:StatsBase.AbstractWeights} = nothing)
    N = length(v1)
    @argcheck(1 < N, DimensionMismatch("1 < length(v1) must hold. Got\nlength(v1) => $N"))
    @argcheck(N == length(v2), DimensionMismatch)
    a, b = calc_pairwise_dists(ce, v1, v2)
    A, B = calc_centred_dists(a, w), calc_centred_dists(b, w)
    return sqrt(calc_dcov2(A, B, w))
end
"""
    cov_distance(ce::DistanceCovariance, X::MatNum,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)

Compute the pairwise distance covariance matrix for all columns in a data matrix using a configured [`DistanceCovariance`](@ref) estimator.

This function computes the distance covariance between each pair of columns in `X`, using the specified distance metric, optional weights, and parallel execution strategy. The resulting matrix is symmetric, with each entry representing the distance covariance between two assets.

# Algorithm

 1. Allocate an ``N \\times N`` matrix, where ``N`` is the number of columns of `X`. The result is indexed by asset and never by observation, so a non-square `X` cannot hide a transposed index.
 2. For each column `j`, and for each column `i` at or below `j`, call [`cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)`](@ref) on the two columns and write the value into both `sigma[i, j]` and `sigma[j, i]`. The estimator's `ex` field runs the outer loop.
 3. Return the symmetric matrix. Its diagonal is the distance standard deviation of each asset, because the pair `(j, j)` is one of the pairs step 2 computes.

# Arguments

  - `ce`: Distance covariance estimator.
  - `X`: Data matrix (observations × assets).
  - `w`: Observation weights, or `nothing` for the unweighted statistic.

# Returns

  - `sigma::Matrix{<:Number}`: Symmetric matrix of pairwise distance covariances.

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

This method overrides the generic covariance fallback, which [`DistanceCovariance`](@ref) cannot use because it carries no variance estimator field. So the diagonal is the **distance** standard deviation of each asset and not the sample standard deviation, and the matrix that [`cor(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref) returns is exactly this matrix rescaled by the square roots of that diagonal.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), which transposes it when `dims` is `2` and refuses any other value.
 2. Resolve the estimator's `w` field against the oriented matrix with [`get_observation_weights`](@ref), giving `nothing` for an unweighted estimator.
 3. Return [`cov_distance(ce::DistanceCovariance, X::MatNum)`](@ref) of the oriented matrix and those weights.

# Arguments

  - `ce`: Distance covariance estimator.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments (currently unused).

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `sigma::Matrix{<:Number}`: Symmetric matrix of pairwise distance covariances.

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
    X = dims_oriented(dims, X)
    w = get_observation_weights(ce.w, X)
    return cov_distance(ce, X, w)
end

export DistanceCovariance
