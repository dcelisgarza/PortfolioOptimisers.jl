"""
$(DocStringExtensions.TYPEDEF)

Measures linear and non-linear codependence from doubly-centred pairwise distance matrices.

The statistic is the distance covariance. Its population value is zero if and only if the two series are independent. `metric`, `args` and `kwargs` configure the pairwise distance. `w` weights the observations, and `ex` selects the parallel execution strategy.

Section 2.2 of [szekely2007](@cite) defines the sample statistics. Definition 4 gives the doubly centred distance matrices. Equation 2.8 gives the squared distance covariance, the mean of their element-wise product with the divisor ``n^{2}``, which is a V-statistic. Definition 5 and Equations 2.9 and 2.10 give the distance correlation, the square root of ``\\mathcal{V}_n^2(X, Y) / \\sqrt{\\mathcal{V}_n^2(X)\\,\\mathcal{V}_n^2(Y)}``. Without weights, the library computes these values. The observation weights `w` and the choice of `metric` are extensions of the library.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DistanceCovariance(;
        metric::Distances.Metric = Distances.Euclidean(),
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        w::Option{<:ObsWeights} = nothing,
        ex::FLoops.Transducers.Executor = ThreadedEx(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> DistanceCovariance

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@wprop` and `@fprop` fields:

  - `w`: Replaced with the incoming [`ObsWeights`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of this `@vprop` field at the selected indices:

  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

## Observation weight parameters

[`obs_weights_view`](@ref) indexes these fields to the selected observations:

  - `w`: Indexed to the selected observations through [`obs_weights_view`](@ref).

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
  - [`Distances.Metric`](https://docs.juliahub.com/General/Distances/stable/)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-ex)
  - [`factory`](@ref)
  - [`obs_weights_view`](@ref)
  - [`cor_distance`](@ref)
  - [`cov_distance`](@ref)

# References

  - $(ref_dict[:szekely2007]) Section 2.2, Definitions 4 and 5, Equations 2.8 to 2.10.
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
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function DistanceCovariance(metric::Distances.Metric, args::Tuple, kwargs::NamedTuple,
                                w::Option{<:ObsWeights}, ex::FLoops.Transducers.Executor,
                                cache::Option{<:AbstractPartialFitState})
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(metric), typeof(args), typeof(kwargs), typeof(w), typeof(ex),
                   typeof(cache)}(metric, args, kwargs, w, ex, cache)
    end
end
function DistanceCovariance(; metric::Distances.Metric = Distances.Euclidean(),
                            args::Tuple = (), kwargs::NamedTuple = (;),
                            w::Option{<:ObsWeights} = nothing,
                            ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                            cache::Option{<:AbstractPartialFitState} = nothing)::DistanceCovariance
    return DistanceCovariance(metric, args, kwargs, w, ex, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`DistanceCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit. That state is not part of the configuration, and it prints under the estimator at each site that renders the estimator. Set `set_show_nothing_fields!(:DistanceCovariance, true)` to render it.

# Arguments

  - `::DistanceCovariance`: The distance covariance estimator. The method reads only its type.

# Returns

  - `fields::Tuple`: The names of the fields to render, `(:metric, :args, :kwargs, :w, :ex)`.

# Related

  - [`DistanceCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::DistanceCovariance) = (:metric, :args, :kwargs, :w, :ex)
"""
    calc_pairwise_dists(ce::DistanceCovariance, v1::VecNum, v2::VecNum) -> (MatNum, MatNum)

Compute the pairwise distance matrix of each of two vectors with the metric of the estimator.

[`cor_distance`](@ref) and [`cov_distance`](@ref) call this helper. It takes no observation weights. The weights apply to the statistic that [`calc_centred_dists`](@ref) and [`calc_dcov2`](@ref) compute, and not to the data that the metric measures.

# Algorithm

 1. Call `Distances.pairwise` with `ce.metric`, `ce.args` and `ce.kwargs` on `v1` and on `v2`. The method does not scale or weight the two vectors.
 2. Return the two matrices as `(D1, D2)`. Each matrix is ``T \\times T`` for series of ``T`` observations. `D1[t, s]` is the distance between observations ``t`` and ``s`` of `v1`.

# Arguments

  - `ce`: The [`DistanceCovariance`](@ref) estimator that holds the metric.
  - `v1`, `v2`: The two data vectors.

# Returns

  - The tuple `(D1, D2)` of the pairwise distance matrices of `v1` and of `v2`.

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

Doubly centre a pairwise distance matrix, with the weight `w` on each observation.

[`cor_distance`](@ref) and [`cov_distance`](@ref) call this helper. The weights apply to the three means. A constant weight vector cancels from each mean, so it gives the unweighted matrix.

# Algorithm

 1. Without weights, take the row means, the column means and the grand mean of `a`.
 2. With weights, take the same three means with the weights `w`. Each mean divides by the sum of the weights that it uses, so the scale of `w` cancels.
 3. Subtract the row means and the column means from `a`. Add the grand mean to get `A`.

# Arguments

  - `a`: The pairwise distance matrix.
  - `w`: The observation weights, or `nothing` for the unweighted statistic.

# Returns

  - `A::MatNum`: The doubly centred distance matrix.

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

[`cor_distance`](@ref) and [`cov_distance`](@ref) call this helper. The weights apply to the two indices of the contraction, so a constant weight vector cancels and gives the unweighted value.

# Algorithm

 1. Without weights, divide the Frobenius inner product of `A` and `B` by the number of entries of `A`.
 2. With weights, contract the element-wise product of `A` and `B` with `w` on its two indices. Divide the result by the square of the sum of the weights.

# Arguments

  - `A`, `B`: The doubly centred distance matrices that [`calc_centred_dists`](@ref) returns.
  - `w`: The observation weights, or `nothing` for the unweighted statistic.

# Returns

  - `dcov2::Number`: The squared distance covariance of the two matrices.

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

Compute the distance correlation of two vectors with a [`DistanceCovariance`](@ref) estimator.

The metric, the `args` and the `kwargs` of `ce` set the pairwise distances, and `w` sets the weights of the observations. A constant series has a distance variance of zero, so its distance correlation with each series is `NaN`.

# Mathematical definition

Let ``a_{ts} = d(v_{1t}, v_{1s})`` and ``b_{ts} = d(v_{2t}, v_{2s})`` be the entries of the two pairwise distance matrices. The doubly centred matrices subtract and add three means, and each mean uses the observation weights:

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

  - ``a_{ts}``, ``b_{ts}``: The pairwise distances between observations ``t`` and ``s``.
  - ``\\bar{a}_{t\\cdot}``: The ``t``-th weighted row mean of ``\\mathbf{a}``.
  - ``\\bar{a}_{\\cdot s}``: The ``s``-th weighted column mean of ``\\mathbf{a}``.
  - ``\\bar{a}_{\\cdot\\cdot}``: The weighted grand mean of ``\\mathbf{a}``.
  - ``A_{ts}``, ``B_{ts}``: The doubly centred pairwise distances.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])

The three means of ``\\mathbf{b}`` have the same form as the three means of ``\\mathbf{a}``.

The squared distance covariances and the distance correlation are:

```math
\\begin{align}
\\widehat{\\mathrm{dCov}}^2(X,X) &= \\frac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{s=1}^{T} w_{t} w_{s} A_{ts} A_{ts}}{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}\\,, \\\\
\\widehat{\\mathrm{dCov}}^2(X,Y) &= \\frac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{s=1}^{T} w_{t} w_{s} A_{ts} B_{ts}}{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}\\,, \\\\
\\widehat{\\mathrm{dCov}}^2(Y,Y) &= \\frac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{s=1}^{T} w_{t} w_{s} B_{ts} B_{ts}}{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}\\,.
\\end{align}
```

Where:

  - ``\\widehat{\\mathrm{dCov}}^2(X,Y)``: The squared distance covariance of ``X`` and ``Y``.

```math
\\begin{align}
\\hat{R}_{\\mathrm{dist}}(X, Y) &= \\frac{\\sqrt{\\widehat{\\mathrm{dCov}}^2(X,Y)}}{\\sqrt{\\sqrt{\\widehat{\\mathrm{dCov}}^2(X,X)} \\cdot \\sqrt{\\widehat{\\mathrm{dCov}}^2(Y,Y)}}}\\,.
\\end{align}
```

Where:

  - ``\\hat{R}_{\\mathrm{dist}}(X, Y)``: The distance correlation of ``X`` and ``Y``.

In each quotient above, the numerator and the denominator have the same degree in the weights, so a constant weight cancels. With ``w_{t} = 1``, the squared distance covariance of ``X`` and ``Y`` is ``\\mathbf{A}:\\mathbf{B} / T^{2}``, the Frobenius inner product ``\\sum_{t,\\,s} A_{ts} B_{ts}`` divided by ``T^{2}``. This is the unweighted statistic, and the other two squared distance covariances have the same form.

# Algorithm

 1. Compute the two pairwise distance matrices ``\\mathbf{a}`` and ``\\mathbf{b}`` with [`calc_pairwise_dists`](@ref). It uses the metric, the `args` and the `kwargs` of `ce`.
 2. Doubly centre each matrix with [`calc_centred_dists`](@ref) to get ``\\mathbf{A}`` and ``\\mathbf{B}``. The three means that it subtracts and adds use the weights.
 3. Contract the three pairs ``(\\mathbf{A}, \\mathbf{A})``, ``(\\mathbf{A}, \\mathbf{B})`` and ``(\\mathbf{B}, \\mathbf{B})`` with [`calc_dcov2`](@ref) to get the three squared distance covariances.
 4. Return ``\\sqrt{\\widehat{\\mathrm{dCov}}^2(X,Y)}`` divided by the square root of the product of the other two square roots.

# Arguments

  - `ce`: The distance covariance estimator.
  - `v1`: The first data vector.
  - `v2`: The second data vector.
  - `w`: The observation weights, or `nothing` for the unweighted statistic.

# Validation

  - `length(v1) == length(v2)`.
  - `length(v1) > 1`.

# Returns

  - `rho::Number`: The distance correlation of `v1` and `v2`. A series that is not constant gives exactly `1.0` against itself. A constant series gives `NaN`.

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

Compute the distance correlation matrix of the columns of a data matrix with a [`DistanceCovariance`](@ref) estimator.

Entry ``(i, j)`` of the symmetric result is the distance correlation of assets ``i`` and ``j``. The metric of `ce` sets the distances, `w` sets the observation weights, and `ce.ex` runs the loop over the columns.

# Algorithm

 1. Allocate an ``N \\times N`` matrix, where ``N`` is the number of columns of `X`. The indices of the result are assets and not observations, so a non-square `X` cannot hide a transposed index.
 2. For each column `j`, and for each column `i` with `i <= j`, call [`cor_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)`](@ref) on the two columns. Write the value into `rho[i, j]` and into `rho[j, i]`. The executor `ce.ex` runs the outer loop.
 3. Return the symmetric matrix. Step 2 computes the pair `(j, j)` too, so the diagonal entry of a column that is not constant is exactly `1.0`. The diagonal entry of a constant column is `NaN`.

# Arguments

  - `ce`: The distance covariance estimator.
  - `X`: The data matrix, `observations × assets`.
  - `w`: The observation weights, or `nothing` for the unweighted statistic.

# Returns

  - `rho::Matrix{<:Number}`: The distance correlation matrix.

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

Compute the distance correlation matrix of the columns of a data matrix with a [`DistanceCovariance`](@ref) estimator.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), which transposes it when `dims` is `2` and refuses any other value.
 2. Refuse a gapped sample with [`assert_finite_sample`](@ref).
 3. Get the observation weights from `ce.w` and the oriented matrix with [`get_observation_weights`](@ref). An unweighted estimator gets `nothing`.
 4. Return [`cor_distance(ce::DistanceCovariance, X::MatNum)`](@ref) of the oriented matrix and those weights.

# Arguments

  - `ce`: The distance covariance estimator.
  - `X`: The data matrix, `observations × assets`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method ignores.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `rho::Matrix{<:Number}`: The symmetric matrix of pairwise distance correlations. The diagonal entry of a column that is not constant is exactly `1.0`, and that of a constant column is `NaN`.

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
    assert_finite_sample(X)
    w = get_observation_weights(ce.w, X)
    return cor_distance(ce, X, w)
end
"""
    cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum,
                 w::Option{<:StatsBase.AbstractWeights} = nothing)

Compute the distance covariance of two vectors with a [`DistanceCovariance`](@ref) estimator.

The metric, the `args` and the `kwargs` of `ce` set the pairwise distances, and `w` sets the weights of the observations.

# Mathematical definition

The doubly centred matrices ``\\mathbf{A}`` and ``\\mathbf{B}`` are those of [`cor_distance`](@ref):

```math
\\begin{align}
\\widehat{\\mathrm{dCov}}(X, Y) &= \\sqrt{\\frac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{s=1}^{T} w_{t} w_{s} A_{ts} B_{ts}}{\\left(\\sum\\limits_{t=1}^{T} w_{t}\\right)^{2}}}\\,.
\\end{align}
```

Where:

  - ``\\widehat{\\mathrm{dCov}}(X, Y)``: The distance covariance of ``X`` and ``Y``.
  - ``A_{ts}``, ``B_{ts}``: The doubly centred pairwise distances.
  - $(math_dict[:w_t_obs])
  - $(math_dict[:T])

With ``w_{t} = 1``, the quotient is ``\\mathbf{A}:\\mathbf{B} / T^{2}``, the Frobenius inner product ``\\sum_{t,\\,s} A_{ts} B_{ts}`` divided by ``T^{2}``. This is the unweighted statistic.

The square root takes no absolute value, and it needs none. The doubly centred V-statistic is non-negative for a metric of strong negative type, and the Euclidean default is such a metric.

# Algorithm

 1. Compute the two pairwise distance matrices ``\\mathbf{a}`` and ``\\mathbf{b}`` with [`calc_pairwise_dists`](@ref). It uses the metric, the `args` and the `kwargs` of `ce`.
 2. Doubly centre each matrix with [`calc_centred_dists`](@ref) to get ``\\mathbf{A}`` and ``\\mathbf{B}``. The three means that it subtracts and adds use the weights.
 3. Contract the pair ``(\\mathbf{A}, \\mathbf{B})`` with [`calc_dcov2`](@ref) to get the squared distance covariance.
 4. Return the square root of that value. Steps 1 and 2 are the steps of [`cor_distance`](@ref). This method computes one of the three contractions, and it computes no ratio.

# Arguments

  - `ce`: The distance covariance estimator.
  - `v1`: The first data vector.
  - `v2`: The second data vector.
  - `w`: The observation weights, or `nothing` for the unweighted statistic.

# Validation

  - `length(v1) == length(v2)`.
  - `length(v1) > 1`.

# Returns

  - `rho::Number`: The distance covariance of `v1` and `v2`. A series against itself gives its distance standard deviation ``\\widehat{\\mathrm{dVar}}^{1/2}(X)``, which is not the sample standard deviation. A constant series gives `0`.

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

Compute the distance covariance matrix of the columns of a data matrix with a [`DistanceCovariance`](@ref) estimator.

Entry ``(i, j)`` of the symmetric result is the distance covariance of assets ``i`` and ``j``. The metric of `ce` sets the distances, `w` sets the observation weights, and `ce.ex` runs the loop over the columns.

# Algorithm

 1. Allocate an ``N \\times N`` matrix, where ``N`` is the number of columns of `X`. The indices of the result are assets and not observations, so a non-square `X` cannot hide a transposed index.
 2. For each column `j`, and for each column `i` with `i <= j`, call [`cov_distance(ce::DistanceCovariance, v1::VecNum, v2::VecNum)`](@ref) on the two columns. Write the value into `sigma[i, j]` and into `sigma[j, i]`. The executor `ce.ex` runs the outer loop.
 3. Return the symmetric matrix. Step 2 computes the pair `(j, j)` too, so the diagonal holds the distance standard deviation of each asset.

# Arguments

  - `ce`: The distance covariance estimator.
  - `X`: The data matrix, `observations × assets`.
  - `w`: The observation weights, or `nothing` for the unweighted statistic.

# Returns

  - `sigma::Matrix{<:Number}`: The symmetric matrix of pairwise distance covariances.

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

Compute the distance covariance matrix of the columns of a data matrix with a [`DistanceCovariance`](@ref) estimator.

This method replaces the generic covariance fallback. [`DistanceCovariance`](@ref) cannot use the fallback, because it has no variance estimator field. So the diagonal is the distance standard deviation of each asset and not the sample standard deviation. The matrix that [`cor(ce::DistanceCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref) returns is this matrix divided by the square roots of its diagonal, row by row and column by column.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), which transposes it when `dims` is `2` and refuses any other value.
 2. Refuse a gapped sample with [`assert_finite_sample`](@ref).
 3. Get the observation weights from `ce.w` and the oriented matrix with [`get_observation_weights`](@ref). An unweighted estimator gets `nothing`.
 4. Return [`cov_distance(ce::DistanceCovariance, X::MatNum)`](@ref) of the oriented matrix and those weights.

# Arguments

  - `ce`: The distance covariance estimator.
  - `X`: The data matrix, `observations × assets`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method ignores.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `sigma::Matrix{<:Number}`: The symmetric matrix of pairwise distance covariances.

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
    assert_finite_sample(X)
    w = get_observation_weights(ce.w, X)
    return cov_distance(ce, X, w)
end

export DistanceCovariance
