"""
$(DocStringExtensions.TYPEDEF)

Measures co-movement in the lower tail, the share of the worst returns of one asset that fall on the same dates as the worst returns of another asset.

The statistic is the empirical lower tail dependence coefficient at the level `alpha`. It measures the dependence of joint drawdowns and of stress scenarios. A correlation over the full sample mixes that dependence with the dependence of the rest of the distribution.

The coefficient is the limit ``\\lambda_L = \\lim_{v \\to 0^{+}} \\Pr[U_1 \\leq v \\mid U_2 \\leq v]`` of the conditional probability that one series is in its lower tail when the other series is in its lower tail. Section 2 and Equation 1 of [luca2011](@cite) state it, and they also state it with the copula ``C`` as ``\\lim_{v \\to 0^{+}} C(v,\\, v) / v``. Definition 1 of [sibuya1960](@cite) gives the dependence function ``\\Omega``, with ``\\Pr(X < x,\\, Y < y) = \\Omega(G, H)\\,G\\,H``, so ``\\lambda_L`` is the limit of ``v\\,\\Omega(v,\\, v)``. That work does not name the coefficient, and it does not state an estimator of it.

The library counts the joint exceedances and divides the count by ``k = \\lceil T \\alpha \\rceil``. This is the rank form of the empirical lower tail copula at ``(1,\\, 1)`` in Equations 12 and 16 of Section 4 of [schmidt2006](@cite). [luca2011](@cite) estimates the coefficient from a fitted Archimedean copula, so it does not state this estimator. `alpha` sets the tail fraction ``k / T``, so `alpha` is the threshold of that estimator.

!!! warning

    The estimate is biased, not only noisy. At a fixed `alpha`, it converges to the ratio ``C(\\alpha,\\, \\alpha) / \\alpha`` as ``T`` grows, and not to the limit ``\\lambda_L``. A Gaussian copula has ``\\lambda_L = 0`` at each correlation below `1`, but at correlation `0.6` and `alpha = 0.05` its ratio is about `0.31`. So a Gaussian sample shows a strong lower tail dependence that its model does not have. To compare a model with an estimate, compare the ratio of the model at the same `alpha`, not its limit. [`lower_tail_dependence`](@ref) gives the ratio at more levels.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LowerTailDependenceCovariance(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        alpha::Number = 0.05,
        ex::FLoops.Transducers.Executor = ThreadedEx(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> LowerTailDependenceCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ve`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ve`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> LowerTailDependenceCovariance()
LowerTailDependenceCovariance
     ve ┼ SimpleVariance
        │          me ┼ SimpleExpectedReturns
        │             │   w ┴ nothing
        │           w ┼ nothing
        │   corrected ┴ Bool: true
  alpha ┼ Float64: 0.05
     ex ┴ Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`SimpleVariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`lower_tail_dependence`](@ref)
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-ex)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:sibuya1960]) Section 1, Definition 1, Equations 1.3 and 1.4.
  - $(ref_dict[:luca2011]) Section 2, Equation 1.
  - $(ref_dict[:schmidt2006]) Section 4, Equations 12 and 16.
"""
@propagatable @concrete struct LowerTailDependenceCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ve])
    """
    @fprop @vprop ve
    """
    $(field_dict[:alpha_ltd])
    """
    alpha
    """
    $(field_dict[:ex])
    """
    ex
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function LowerTailDependenceCovariance(ve::AbstractVarianceEstimator, alpha::Number,
                                           ex::FLoops.Transducers.Executor,
                                           cache::Option{<:AbstractPartialFitState})
        assert_unit_interval(alpha, :alpha)
        return new{typeof(ve), typeof(alpha), typeof(ex), typeof(cache)}(ve, alpha, ex,
                                                                         cache)
    end
end
function LowerTailDependenceCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                                       alpha::Number = 0.05,
                                       ex::FLoops.Transducers.Executor = FLoops.ThreadedEx(),
                                       cache::Option{<:AbstractPartialFitState} = nothing)::LowerTailDependenceCovariance
    return LowerTailDependenceCovariance(ve, alpha, ex, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`LowerTailDependenceCovariance`](@ref) except `cache`.

A `cache` holds the state of an incremental fit. That state is not part of the configuration, and it prints under the estimator at each site that renders the estimator. Set `set_show_nothing_fields!(:LowerTailDependenceCovariance, true)` to render it.

# Arguments

  - `::LowerTailDependenceCovariance`: The lower tail dependence covariance estimator. The method reads only its type.

# Returns

  - `fields::Tuple`: The names of the fields to render, `(:ve, :alpha, :ex)`.

# Related

  - [`LowerTailDependenceCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::LowerTailDependenceCovariance) = (:ve, :alpha, :ex)
"""
    lower_tail_dependence(X::MatNum, alpha::Number = 0.05,
                          ex::FLoops.Transducers.Executor = FLoops.SequentialEx())

Compute the lower tail dependence matrix of a matrix of asset returns.

The lower tail dependence of two assets is the probability that the returns of the two assets are in their lower tails at the same time, at the quantile level `alpha`. The function estimates it for each pair of columns of `X`.

# Mathematical definition

For a quantile level ``\\alpha \\in (0,1)`` and ``k = \\lceil T \\alpha \\rceil``, let ``\\hat{q}_i`` be the empirical ``\\alpha``-quantile of asset ``i``, which is the ``k``-th order statistic. The estimate of the lower tail dependence of assets ``i`` and ``j`` is:

```math
\\begin{align}
\\hat{\\lambda}_{ij} &= \\frac{1}{k} \\sum_{t=1}^{T} \\mathbf{1}\\left[x_{ti} \\leq \\hat{q}_i \\text{ and } x_{tj} \\leq \\hat{q}_j\\right]\\,.
\\end{align}
```

Where:

  - ``\\hat{\\lambda}_{ij}``: The estimate of the lower tail dependence of assets ``i`` and ``j``.
  - $(math_dict[:T])
  - $(math_dict[:alpha_rm])
  - ``k = \\lceil T \\alpha \\rceil``: The number of observations in the lower tail.
  - ``x_{ti}``: The return of asset ``i`` at time ``t``.
  - ``\\hat{q}_i``: The empirical ``\\alpha``-quantile of asset ``i``.

The function clamps each entry of the symmetric result to ``[\\sqrt{\\varepsilon},\\, 1]``, where ``\\varepsilon`` is the machine epsilon of `eltype(X)`. The floor is not zero. A pair whose tails never coincide gets `1.4901161193847656e-8` for a `Float64` input, so a caller can divide by an entry or take its logarithm.

!!! warning

    The estimate is biased, not only noisy, and the bias decreases slowly as `alpha` decreases. At a fixed `alpha`, the joint count over ``k`` converges to ``C(\\alpha,\\, \\alpha) / \\alpha`` as ``T`` grows, where ``C`` is the copula of the pair. That ratio is the conditional probability at the level `alpha`, and not its limit ``\\lambda_L`` at zero. The multivariate normal distribution is lower tail independent, so a Gaussian copula with a correlation ``\\rho < 1`` has ``\\lambda_L = 0``. But its ratio is far from zero at the levels that a sample of returns can support:

    | ``\\rho`` | `alpha = 0.05` | `alpha = 0.01` | `alpha = 0.001` |
    |:--------- | --------------:| --------------:| ---------------:|
    | `0.3`     | `0.143`        | `0.056`        | `0.015`         |
    | `0.6`     | `0.310`        | `0.188`        | `0.095`         |
    | `0.8`     | `0.495`        | `0.377`        | `0.263`         |

    A smaller `alpha` decreases the bias, but it also decreases ``k``, so the variance of the estimate increases. The choice of ``k`` is thus a variance-bias problem, and the finite-sample bias increases with the correlation. Do not read a positive estimate as evidence of tail dependence. A Gaussian sample with correlation `0.6` gives about `0.31` at `alpha = 0.05`. To compare a model with an estimate, compare the ratio of the model at the same `alpha`, not its limit.

# Algorithm

 1. Set ``k = \\lceil T \\alpha \\rceil``, the number of observations in the lower tail. Allocate the result as a matrix of zeros. The allocation is `zeros` and not `undef`, so the caller gets zeros and not uninitialised memory when `k` is not positive.
 2. If `k` is not positive, return the zero matrix. The `# Validation` section states the cases that get to this branch.
 3. Copy `X`. Partially sort each column of the copy about position `k`. Position `k` of the sorted column is then the ``k``-th order statistic, which is the empirical ``\\alpha``-quantile of that asset.
 4. Fill the boolean mask one column at a time. Entry `(t, j)` is true when ``x_{tj}`` is at or below the quantile of step 3. The method fills all of the mask before step 5. Iteration `j` of the threaded loop of step 5 reads each column `i <= j`, so a mask that the loop fills itself can be incomplete when an iteration reads it.
 5. For each pair ``(i, j)`` with ``i \\leq j``, count the observations at which the two mask entries are true. Divide the count by `k`, and clamp the ratio to ``[\\sqrt{\\varepsilon},\\, 1]``. Write the clamped value into `rho[i, j]` and into `rho[j, i]`. The executor `ex` runs this loop.

# Arguments

  - `X`: The data matrix of asset returns, `observations × assets`.
  - `alpha`: The quantile level of the lower tail.
  - `ex`: The parallel execution strategy.

# Validation

  - `ceil(Int, T * alpha) > 0`. If the count is not positive, the function returns a zero matrix and not an estimate. This happens when `alpha <= 0` or when `X` has no observation. A small positive `alpha` does not get to this branch, because `ceil` of a positive real is at least `1`. So `alpha = nextfloat(0.0)` on a matrix of one or more observations selects one observation.
  - `ceil(Int, T * alpha) <= T`. The function does not check it. An `alpha > 1` makes `k` larger than `T`, and `partialsort!` raises a `BoundsError`.
  - [`LowerTailDependenceCovariance`](@ref) refuses an `alpha` outside ``(0, 1)`` at construction, because [`assert_unit_interval`](@ref) requires `0 < alpha < 1`. So only a direct call gets to the two cases above.

# Returns

  - `rho::Matrix{<:Number}`: The symmetric matrix of lower tail dependence coefficients. `rho[i, j]` is the estimate for assets `i` and `j`. If `k` is positive, the diagonal is exactly `1` and each entry is in ``[\\sqrt{\\varepsilon},\\, 1]``. If `k` is not positive, each entry is `0`.

# Related

  - [`LowerTailDependenceCovariance`](@ref)
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-ex)

# References

  - $(ref_dict[:schmidt2006]) Section 2, the tail independence of the multivariate normal distribution. Section 4, the variance-bias problem of the choice of ``k``. Section 7.1, Table 1, the finite-sample bias.
"""
function lower_tail_dependence(X::MatNum, alpha::Number = 0.05,
                               ex::FLoops.Transducers.Executor = FLoops.SequentialEx())
    T, N = size(X)
    k = ceil(Int, T * alpha)
    # `zeros`, not `undef`: when `alpha <= 0` or `T == 0`, `k` is not positive and the loop
    # below never runs, so an uninitialised matrix would reach the caller as an estimate.
    rho = zeros(eltype(X), N, N)
    if k > 0
        Xs = copy(X)
        mask = falses(T, N)
        for i in axes(Xs, 2)
            #! Use the weighted ValueatRisk formulation to account for observation weights
            partialsort!(view(Xs, :, i), k)
        end
        mv = sqrt(eps(eltype(X)))
        # The mask must be complete before the pair counts: iteration j reads
        # column i < j, so filling it inside the threaded loop is a race.
        for j in axes(X, 2)
            mask[:, j] .= view(X, :, j) .<= Xs[k, j]
        end
        FLoops.@floop ex for j in axes(X, 2)
            for i in 1:j
                ltd = count(view(mask, :, i) .&& view(mask, :, j)) / k
                rho[j, i] = rho[i, j] = clamp(ltd, mv, one(eltype(X)))
            end
        end
    end
    return rho
end
"""
    Statistics.cor(ce::LowerTailDependenceCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the lower tail dependence correlation matrix with a [`LowerTailDependenceCovariance`](@ref) estimator.

The quantile level `ce.alpha` and the executor `ce.ex` configure the computation. Entry ``(i, j)`` is the probability that assets ``i`` and ``j`` have returns in their lower tails at the same time. The generic covariance fallback reads the `ve` field of the estimator, and this method does not read it.

# Algorithm

 1. Orient `X` with [`dims_oriented`](@ref), which transposes it when `dims` is `2` and refuses any other value.
 2. Refuse a gapped sample with [`assert_finite_sample`](@ref).
 3. Return [`lower_tail_dependence`](@ref) of the oriented matrix, with the `alpha` and the executor `ex` of the estimator.

# Arguments

  - `ce`: The lower tail dependence covariance estimator.
  - `X`: The data matrix of asset returns, `observations × assets`.
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments, which the method ignores.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `rho::Matrix{<:Number}`: The symmetric matrix of lower tail dependence coefficients.

# Examples

```jldoctest
julia> ce = LowerTailDependenceCovariance();

julia> X = [0.01 0.02; 0.03 0.04; 0.02 0.03];

julia> cor(ce, X)
2×2 Matrix{Float64}:
 1.0  1.0
 1.0  1.0
```

# Related

  - [`LowerTailDependenceCovariance`](@ref)
  - [`lower_tail_dependence`](@ref)
"""
function Statistics.cor(ce::LowerTailDependenceCovariance, X::MatNum; dims::Int = 1,
                        kwargs...)
    X = dims_oriented(dims, X)
    assert_finite_sample(X)
    return lower_tail_dependence(X, ce.alpha, ce.ex)
end

export LowerTailDependenceCovariance
