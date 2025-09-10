"""
```julia
struct LTDCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
    ve::T1
    alpha::T2
    threads::T3
end
```

Lower tail dependence covariance estimator.

`LTDCovariance` implements a robust covariance estimator based on lower tail dependence, which measures the co-movement of asset returns in the lower quantiles (i.e., during joint drawdowns or adverse events). This estimator is particularly useful for capturing dependence structures relevant to risk management and stress scenarios.

# Fields

  - `ve`: Variance estimator used to compute marginal standard deviations.
  - `alpha`: Quantile level for the 5% lower tail.
  - `threads`: Parallel execution strategy.

# Constructor

    LTDCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                    alpha::Real = 0.05,
                    threads::FLoops.Transducers.Executor = ThreadedEx())

Keyword arguments correspond to the fields above.

## Validation

  - `0 < alpha < 1`.

# Examples

```jldoctest
julia> ce = LTDCovariance()
LTDCovariance
       ve | SimpleVariance
          |          me | SimpleExpectedReturns
          |             |   w | nothing
          |           w | nothing
          |   corrected | Bool: true
    alpha | Float64: 0.05
  threads | Transducers.ThreadedEx{@NamedTuple{}}: Transducers.ThreadedEx()
```

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`SimpleVariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-executor)
"""
struct LTDCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
    ve::T1
    alpha::T2
    threads::T3
end
function LTDCovariance(; ve::AbstractVarianceEstimator = SimpleVariance(),
                       alpha::Real = 0.05,
                       threads::FLoops.Transducers.Executor = ThreadedEx())
    @argcheck(zero(alpha) < alpha < one(alpha))
    return LTDCovariance(ve, alpha, threads)
end
function factory(ce::LTDCovariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return LTDCovariance(; ve = factory(ce.ve, w), alpha = ce.alpha, threads = ce.threads)
end

"""
```julia
lower_tail_dependence(X::AbstractMatrix; alpha::Real = 0.05,
                      threads::FLoops.Transducers.Executor = SequentialEx())
```

Compute the lower tail dependence matrix for a set of asset returns.

The lower tail dependence (LTD) between two assets quantifies the probability that both assets experience returns in their respective lower tails (i.e., joint drawdowns or adverse events), given a specified quantile level `alpha`. This function estimates the LTD matrix for all pairs of assets in the input matrix `X`, which is particularly useful for risk management and stress testing.

# Arguments

  - `X`: Data matrix of asset returns (observations × assets).
  - `alpha`: Quantile level for the lower tail.
  - `threads`: Parallel execution strategy.

# Returns

  - `rho::Matrix{<:Real}`: Symmetric matrix of lower tail dependence coefficients, where `rho[i, j]` is the estimated LTD between assets `i` and `j`.

# Details

For each pair of assets `(i, j)`, the LTD is estimated as the proportion of observations where both asset `i` and asset `j` have returns less than or equal to their respective empirical `alpha`-quantiles, divided by the number of observations in the lower tail (`ceil(Int, T * alpha)`, where `T` is the number of observations).

The resulting matrix is symmetric and all values are clamped to `[0, 1]`.

# Related

  - [`LTDCovariance`](@ref)
  - [`FLoops.Transducers.Executor`](https://juliafolds2.github.io/FLoops.jl/dev/tutorials/parallel/#tutorials-executor)
"""
function lower_tail_dependence(X::AbstractMatrix, alpha::Real = 0.05,
                               threads::FLoops.Transducers.Executor = SequentialEx())
    T, N = size(X)
    k = ceil(Int, T * alpha)
    rho = Matrix{eltype(X)}(undef, N, N)
    if k > 0
        let mv = sqrt(eps(eltype(X)))
            @floop threads for j in axes(X, 2)
                xj = view(X, :, j)
                v = sort(xj)[k]
                maskj = xj .<= v
                for i in 1:j
                    xi = view(X, :, i)
                    u = sort(xi)[k]
                    ltd = sum(xi .<= u .&& maskj) / k
                    rho[j, i] = rho[i, j] = clamp(ltd, mv, one(eltype(X)))
                end
            end
        end
    end
    return rho
end

"""
```julia
cor(ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the lower tail dependence correlation matrix using a [`LTDCovariance`](@ref) estimator.

This method computes the lower tail dependence (LTD) correlation matrix for the input data matrix `X` using the quantile level and parallel execution strategy specified in `ce`. The LTD correlation quantifies the probability that pairs of assets experience joint drawdowns or adverse events, as measured by their co-movement in the lower tail.

# Arguments

  - `ce`: Lower tail dependence covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `rho::Matrix{<:Real}`: Symmetric matrix of lower tail dependence correlation coefficients.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`LTDCovariance`](@ref)
  - [`lower_tail_dependence`](@ref)
"""
function Statistics.cor(ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    return lower_tail_dependence(X, ce.alpha, ce.threads)
end
"""
```julia
cov(ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
```

Compute the lower tail dependence covariance matrix using a [`LTDCovariance`](@ref) estimator.

This method computes the lower tail dependence (LTD) covariance matrix for the input data matrix `X` using the quantile level and parallel execution strategy specified in `ce`. The LTD covariance focuses on the co-movement of asset returns in the lower tail, making it robust to extreme events and particularly relevant for risk-sensitive applications.

# Arguments

  - `ce`: Lower tail dependence covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the variance estimator.

# Returns

  - `sigma::Matrix{<:Real}`: Symmetric matrix of lower tail dependence covariances.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`LTDCovariance`](@ref)
  - [`lower_tail_dependence`](@ref)
"""
function Statistics.cov(ce::LTDCovariance, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    std_vec = std(ce.ve, X; dims = 1, kwargs...)
    return lower_tail_dependence(X, ce.alpha, ce.threads) ⊙ (std_vec ⊗ std_vec)
end

export LTDCovariance
