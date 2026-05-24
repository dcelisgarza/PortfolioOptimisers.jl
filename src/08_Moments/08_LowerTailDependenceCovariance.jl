"""
$(DocStringExtensions.TYPEDEF)

Lower tail dependence covariance estimator.

`LowerTailDependenceCovariance` implements a robust covariance estimator based on lower tail dependence, which measures the co-movement of asset returns in the lower quantiles (i.e., during joint drawdowns or adverse events). This estimator is particularly useful for capturing dependence structures relevant to risk management and stress scenarios.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LowerTailDependenceCovariance(;
        ve::AbstractVarianceEstimator = SimpleVariance(),
        alpha::Number = 0.05,
        ex::FLoops.Transducers.Executor = ThreadedEx()
    ) -> LowerTailDependenceCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:alpha])

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
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-ex)
"""
@concrete struct LowerTailDependenceCovariance <: AbstractCovarianceEstimator
    "$(field_dict[:ve])"
    ve
    "$(field_dict[:alpha])"
    alpha
    "$(field_dict[:ex])"
    ex
    function LowerTailDependenceCovariance(ve::AbstractVarianceEstimator, alpha::Number,
                                           ex::FLoops.Transducers.Executor)
        @argcheck(zero(alpha) < alpha < one(alpha),
                  DomainError("0 < alpha < 1 must hold. Got\nalpha => $alpha"))
        return new{typeof(ve), typeof(alpha), typeof(ex)}(ve, alpha, ex)
    end
end
function LowerTailDependenceCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                                       alpha::Number = 0.05,
                                       ex::FLoops.Transducers.Executor = FLoops.ThreadedEx())::LowerTailDependenceCovariance
    return LowerTailDependenceCovariance(ve, alpha, ex)
end
"""
    factory(ce::LowerTailDependenceCovariance, w::ObsWeights) -> LowerTailDependenceCovariance

Return a new [`LowerTailDependenceCovariance`](@ref) estimator with observation weights `w` applied to the underlying variance estimator.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Related

  - [`LowerTailDependenceCovariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::LowerTailDependenceCovariance,
                 w::ObsWeights)::LowerTailDependenceCovariance
    return LowerTailDependenceCovariance(; ve = factory(ce.ve, w), alpha = ce.alpha,
                                         ex = ce.ex)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the covariance estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:ce])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:cev])

# Related

  - [`LowerTailDependenceCovariance`](@ref)
"""
function moment_view(ce::LowerTailDependenceCovariance, i)::LowerTailDependenceCovariance
    return LowerTailDependenceCovariance(; ve = moment_view(ce.ve, i), alpha = ce.alpha,
                                         ex = ce.ex)
end
"""
    lower_tail_dependence(X::MatNum; alpha::Number = 0.05,
                          ex::FLoops.Transducers.Executor = SequentialEx())

Compute the lower tail dependence matrix for a set of asset returns.

The lower tail dependence (LTD) between two assets quantifies the probability that both assets experience returns in their respective lower tails (i.e., joint drawdowns or adverse events), given a specified quantile level `alpha`. This function estimates the LTD matrix for all pairs of assets in the input matrix `X`, which is particularly useful for risk management and stress testing.

# Mathematical definition

For a quantile level ``\\alpha \\in (0,1)`` and ``k = \\lceil T \\alpha \\rceil``, let ``\\hat{q}_i`` denote the empirical ``\\alpha``-quantile of asset ``i`` (the ``k``-th order statistic). The lower tail dependence between assets ``i`` and ``j`` is estimated as:

```math
\\hat{\\lambda}_{ij} = \\frac{1}{k} \\sum_{t=1}^{T} \\mathbf{1}\\left[x_{ti} \\leq \\hat{q}_i \\text{ and } x_{tj} \\leq \\hat{q}_j\\right]
```

The resulting matrix is symmetric with entries clamped to ``[0,\\, 1]``.

Where:

  - ``T``: Number of observations.
  - ``\\alpha``: Quantile level for the lower tail (e.g., ``0.05``).
  - ``k = \\lceil T \\alpha \\rceil``: Number of observations in the lower tail.
  - ``x_{ti}``: Return of asset ``i`` at time ``t``.
  - ``\\hat{q}_i``: Empirical ``\\alpha``-quantile of asset ``i``.

# Arguments

  - `X`: Data matrix of asset returns (observations × assets).
  - `alpha`: Quantile level for the lower tail.
  - `ex`: Parallel execution strategy.

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of lower tail dependence coefficients, where `rho[i, j]` is the estimated LTD between assets `i` and `j`.

# Details

For each pair of assets `(i, j)`, the LTD is estimated as the proportion of observations where both asset `i` and asset `j` have returns less than or equal to their respective empirical `alpha`-quantiles, divided by the number of observations in the lower tail (`ceil(Int, T * alpha)`, where `T` is the number of observations).

The resulting matrix is symmetric and all values are clamped to `[0, 1]`.

# Related

  - [`LowerTailDependenceCovariance`](@ref)
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-ex)
"""
function lower_tail_dependence(X::MatNum, alpha::Number = 0.05,
                               ex::FLoops.Transducers.Executor = FLoops.SequentialEx())
    T, N = size(X)
    k = ceil(Int, T * alpha)
    rho = Matrix{eltype(X)}(undef, N, N)
    if k > 0
        Xs = copy(X)
        mask = falses(T, N)
        for i in axes(Xs, 2)
            #! Use the weighted ValueatRisk formulation to account for observation weights
            partialsort!(view(Xs, :, i), k)
        end
        mv = sqrt(eps(eltype(X)))
        FLoops.@floop ex for j in axes(X, 2)
            xj = view(X, :, j)
            mask[:, j] .= xj .<= Xs[k, j]
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

Compute the lower tail dependence correlation matrix using a [`LowerTailDependenceCovariance`](@ref) estimator.

This method computes the lower tail dependence (LTD) correlation matrix for the input data matrix `X` using the quantile level and parallel execution strategy specified in `ce`. The LTD correlation quantifies the probability that pairs of assets experience joint drawdowns or adverse events, as measured by their co-movement in the lower tail.

# Arguments

  - `ce`: Lower tail dependence covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `rho::Matrix{<:Number}`: Symmetric matrix of lower tail dependence correlation coefficients.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`LowerTailDependenceCovariance`](@ref)
  - [`lower_tail_dependence`](@ref)
"""
function Statistics.cor(ce::LowerTailDependenceCovariance, X::MatNum; dims::Int = 1,
                        kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return lower_tail_dependence(X, ce.alpha, ce.ex)
end
"""
    Statistics.cov(ce::LowerTailDependenceCovariance, X::MatNum; dims::Int = 1, kwargs...)

Compute the lower tail dependence covariance matrix using a [`LowerTailDependenceCovariance`](@ref) estimator.

This method computes the lower tail dependence (LTD) covariance matrix for the input data matrix `X` using the quantile level and parallel execution strategy specified in `ce`. The LTD covariance focuses on the co-movement of asset returns in the lower tail, making it robust to extreme events and particularly relevant for risk-sensitive applications.

# Arguments

  - `ce`: Lower tail dependence covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the variance estimator.

# Returns

  - `sigma::Matrix{<:Number}`: Symmetric matrix of lower tail dependence covariances.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`LowerTailDependenceCovariance`](@ref)
  - [`lower_tail_dependence`](@ref)
"""
function Statistics.cov(ce::LowerTailDependenceCovariance, X::MatNum; dims::Int = 1,
                        kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sd = Statistics.std(ce.ve, X; dims = 1, kwargs...)
    sigma = lower_tail_dependence(X, ce.alpha, ce.ex)
    return StatsBase.cor2cov!(sigma, sd)
end

export LowerTailDependenceCovariance
