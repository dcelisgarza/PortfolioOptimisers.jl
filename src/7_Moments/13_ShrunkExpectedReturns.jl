"""
    abstract type AbstractShrunkExpectedReturnsEstimator <: AbstractExpectedReturnsEstimator end

Abstract supertype for all shrunk expected returns estimators in PortfolioOptimisers.jl.

All concrete types implementing shrinkage-based expected returns estimation algorithms should subtype `AbstractShrunkExpectedReturnsEstimator`. This enables a consistent interface for shrinkage estimators throughout the package.

# Related

  - [`ShrunkExpectedReturns`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsEstimator <: AbstractExpectedReturnsEstimator end

"""
    abstract type AbstractShrunkExpectedReturnsAlgorithm <: AbstractExpectedReturnsAlgorithm end

Abstract supertype for all shrinkage algorithms for expected returns estimation.

All concrete types implementing specific shrinkage algorithms (e.g., James-Stein, Bayes-Stein) should subtype `AbstractShrunkExpectedReturnsAlgorithm`. This enables flexible extension and dispatch of shrinkage routines.

# Related

  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`AbstractExpectedReturnsAlgorithm`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsAlgorithm <: AbstractExpectedReturnsAlgorithm end

"""
    abstract type AbstractShrunkExpectedReturnsTarget <: AbstractExpectedReturnsAlgorithm end

Abstract supertype for all shrinkage targets used in expected returns estimation.

Concrete types implementing specific shrinkage targets (e.g., grand mean, volatility-weighted mean) should subtype `AbstractShrunkExpectedReturnsTarget`. This enables modular selection of shrinkage targets in shrinkage algorithms.

# Related

  - [`GrandMean`](@ref)
  - [`VolatilityWeighted`](@ref)
  - [`MeanSquareError`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsTarget <: AbstractExpectedReturnsAlgorithm end

"""
    struct GrandMean <: AbstractShrunkExpectedReturnsTarget end

Shrinkage target representing the grand mean of expected returns.

`GrandMean` computes the shrinkage target as the mean of all asset expected returns, resulting in a vector where each element is the same grand mean value. This is commonly used in shrinkage estimators to reduce estimation error by pulling individual expected returns toward the overall average.

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
struct GrandMean <: AbstractShrunkExpectedReturnsTarget end

"""
    struct VolatilityWeighted <: AbstractShrunkExpectedReturnsTarget end

Shrinkage target representing the volatility-weighted mean of expected returns.

`VolatilityWeighted` computes the shrinkage target as a weighted mean of expected returns, where weights are inversely proportional to asset volatility (from the inverse covariance matrix). This approach accounts for differences in asset risk when estimating the shrinkage target.

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
struct VolatilityWeighted <: AbstractShrunkExpectedReturnsTarget end

"""
    struct MeanSquareError <: AbstractShrunkExpectedReturnsTarget end

Shrinkage target representing the mean squared error of expected returns.

`MeanSquareError` computes the shrinkage target as the trace of the covariance matrix divided by the number of observations, resulting in a vector where each element is the same value. This target is useful for certain shrinkage estimators that minimize mean squared error.

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
struct MeanSquareError <: AbstractShrunkExpectedReturnsTarget end

"""
    struct JamesStein{T1 <: AbstractShrunkExpectedReturnsTarget} <: AbstractShrunkExpectedReturnsAlgorithm
        target::T1
    end

Shrinkage algorithm implementing the James-Stein estimator for expected returns.

`JamesStein` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean). The estimator reduces estimation error, especially in high-dimensional settings.

# Fields

  - `target::AbstractShrunkExpectedReturnsTarget`: The shrinkage target type.

# Constructor

    JamesStein(; target::AbstractShrunkExpectedReturnsTarget = GrandMean())

Construct a `JamesStein` shrinkage algorithm with the specified target.

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
"""
struct JamesStein{T1 <: AbstractShrunkExpectedReturnsTarget} <:
       AbstractShrunkExpectedReturnsAlgorithm
    target::T1
end
"""
    JamesStein(; target::AbstractShrunkExpectedReturnsTarget = GrandMean())

Construct a [`JamesStein`](@ref) shrinkage algorithm for expected returns estimation.

# Arguments

  - `target::AbstractShrunkExpectedReturnsTarget`: The shrinkage target.

# Returns

  - `JamesStein`: Configured James-Stein shrinkage algorithm.

# Examples

```jldoctest
julia> JamesStein()
JamesStein
  target | GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
"""
function JamesStein(; target::AbstractShrunkExpectedReturnsTarget = GrandMean())
    return JamesStein{typeof(target)}(target)
end

"""
    struct BayesStein{T1 <: AbstractShrunkExpectedReturnsTarget} <: AbstractShrunkExpectedReturnsAlgorithm
        target::T1
    end

Shrinkage algorithm implementing the Bayes-Stein estimator for expected returns.

`BayesStein` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean) using Bayesian principles. This estimator is useful for reducing estimation error, especially when sample sizes are small.

# Fields

  - `target::AbstractShrunkExpectedReturnsTarget`: The shrinkage target type.

# Constructor

    BayesStein(; target::AbstractShrunkExpectedReturnsTarget = GrandMean())

Construct a `BayesStein` shrinkage algorithm with the specified target.

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
"""
struct BayesStein{T1 <: AbstractShrunkExpectedReturnsTarget} <:
       AbstractShrunkExpectedReturnsAlgorithm
    target::T1
end
"""
    BayesStein(; target::AbstractShrunkExpectedReturnsTarget = GrandMean())

Construct a [`BayesStein`](@ref) shrinkage algorithm for expected returns estimation.

# Arguments

  - `target::AbstractShrunkExpectedReturnsTarget`: The shrinkage target.

# Returns

  - `BayesStein`: Configured Bayes-Stein shrinkage algorithm.

# Examples

```jldoctest
julia> BayesStein()
BayesStein
  target | GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
"""
function BayesStein(; target::AbstractShrunkExpectedReturnsTarget = GrandMean())
    return BayesStein{typeof(target)}(target)
end

"""
    struct BodnarOkhrinParolya{T1 <: AbstractShrunkExpectedReturnsTarget} <: AbstractShrunkExpectedReturnsAlgorithm
        target::T1
    end

Shrinkage algorithm implementing the Bodnar-Okhrin-Parolya estimator for expected returns.

`BodnarOkhrinParolya` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean) using the Bodnar-Okhrin-Parolya approach. This estimator is designed for robust estimation in high-dimensional settings.

# Fields

  - `target::AbstractShrunkExpectedReturnsTarget`: The shrinkage target type.

# Constructor

    BodnarOkhrinParolya(; target::AbstractShrunkExpectedReturnsTarget = GrandMean())

Construct a `BodnarOkhrinParolya` shrinkage algorithm with the specified target.

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
"""
struct BodnarOkhrinParolya{T1 <: AbstractShrunkExpectedReturnsTarget} <:
       AbstractShrunkExpectedReturnsAlgorithm
    target::T1
end
"""
    BodnarOkhrinParolya(; target::AbstractShrunkExpectedReturnsTarget = GrandMean())

Construct a [`BodnarOkhrinParolya`](@ref) shrinkage algorithm for expected returns estimation.

# Arguments

  - `target::AbstractShrunkExpectedReturnsTarget`: The shrinkage target.

# Returns

  - `BodnarOkhrinParolya`: Configured Bodnar-Okhrin-Parolya shrinkage algorithm.

# Examples

```jldoctest
julia> BodnarOkhrinParolya()
BodnarOkhrinParolya
  target | GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
"""
function BodnarOkhrinParolya(; target::AbstractShrunkExpectedReturnsTarget = GrandMean())
    return BodnarOkhrinParolya{typeof(target)}(target)
end

"""
    struct ShrunkExpectedReturns{T1 <: AbstractExpectedReturnsEstimator,
                                 T2 <: StatsBase.CovarianceEstimator,
                                 T3 <: AbstractShrunkExpectedReturnsAlgorithm} <: AbstractShrunkExpectedReturnsEstimator
        me::T1
        ce::T2
        alg::T3
    end

Container type for shrinkage-based expected returns estimators.

`ShrunkExpectedReturns` encapsulates all components required for shrinkage estimation of expected returns, including the mean estimator, covariance estimator, and shrinkage algorithm. This enables modular and extensible workflows for robust expected returns estimation using shrinkage techniques.

# Fields

  - `me::AbstractExpectedReturnsEstimator`: Mean estimator for expected returns.
  - `ce::StatsBase.CovarianceEstimator`: Covariance estimator.
  - `alg::AbstractShrunkExpectedReturnsAlgorithm`: Shrinkage algorithm (e.g., James-Stein, Bayes-Stein).

# Constructor

    ShrunkExpectedReturns(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                           ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                           alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein())

Construct a `ShrunkExpectedReturns` estimator with the specified mean estimator, covariance estimator, and shrinkage algorithm.

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
"""
struct ShrunkExpectedReturns{T1 <: AbstractExpectedReturnsEstimator,
                             T2 <: StatsBase.CovarianceEstimator,
                             T3 <: AbstractShrunkExpectedReturnsAlgorithm} <:
       AbstractShrunkExpectedReturnsEstimator
    me::T1
    ce::T2
    alg::T3
end
"""
    ShrunkExpectedReturns(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                           ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                           alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein())

Construct a [`ShrunkExpectedReturns`](@ref) estimator for shrinkage-based expected returns estimation.

# Arguments

  - `me::AbstractExpectedReturnsEstimator`: Mean estimator for expected returns.
  - `ce::StatsBase.CovarianceEstimator`: Covariance estimator.
  - `alg::AbstractShrunkExpectedReturnsAlgorithm`: Shrinkage algorithm.

# Returns

  - `ShrunkExpectedReturns`: Configured shrinkage-based expected returns estimator.

# Examples

```jldoctest
julia> ShrunkExpectedReturns()
ShrunkExpectedReturns
   me | SimpleExpectedReturns
      |   w | nothing
   ce | PortfolioOptimisersCovariance
      |   ce | Covariance
      |      |    me | SimpleExpectedReturns
      |      |       |   w | nothing
      |      |    ce | GeneralWeightedCovariance
      |      |       |   ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      |      |       |    w | nothing
      |      |   alg | Full()
      |   mp | DefaultMatrixProcessing
      |      |       pdm | PosdefEstimator
      |      |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
      |      |   denoise | nothing
      |      |    detone | nothing
      |      |       alg | nothing
  alg | JamesStein
      |   target | GrandMean()
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
function ShrunkExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein())
    return ShrunkExpectedReturns{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
end

"""
    target_mean(::GrandMean, mu, sigma; kwargs...)
    target_mean(::VolatilityWeighted, mu, sigma; isigma = nothing, kwargs...)
    target_mean(::MeanSquareError, mu, sigma; T, kwargs...)

Compute the shrinkage target vector for expected returns estimation.

`target_mean` calculates the target vector toward which expected returns are shrunk, based on the specified shrinkage target type. This function is used internally by shrinkage estimators such as James-Stein, Bayes-Stein, and Bodnar-Okhrin-Parolya.

# Arguments

  - `target::GrandMean`: Returns a vector filled with the mean of `mu`.
  - `target::VolatilityWeighted`: Returns a vector filled with the volatility-weighted mean of `mu`, using the inverse covariance matrix.
  - `target::MeanSquareError`: Returns a vector filled with the trace of `sigma` divided by `T`.
  - `mu::AbstractArray`: Vector of expected returns.
  - `sigma::AbstractMatrix`: Covariance matrix of asset returns.
  - `kwargs...`: Additional keyword arguments, such as `T` (number of observations) or `isigma` (inverse covariance matrix).

# Returns

  - `b::AbstractArray`: Target vector for shrinkage estimation.

# Methods

  - `target_mean(::GrandMean, mu, sigma; kwargs...)`: Returns a vector filled with the grand mean of `mu`.
  - `target_mean(::VolatilityWeighted, mu, sigma; isigma = nothing, kwargs...)`: Returns a vector filled with the volatility-weighted mean of `mu`, using the inverse covariance matrix.
  - `target_mean(::MeanSquareError, mu, sigma; T, kwargs...)`: Returns a vector filled with the trace of `sigma` divided by `T`.

# Related

  - [`GrandMean`](@ref)
  - [`VolatilityWeighted`](@ref)
  - [`MeanSquareError`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
function target_mean(::GrandMean, mu::AbstractArray, sigma::AbstractMatrix; kwargs...)
    val = mean(mu)
    return range(; start = val, stop = val, length = length(mu))
end
function target_mean(::VolatilityWeighted, mu::AbstractArray, sigma::AbstractMatrix;
                     isigma = nothing, kwargs...)
    if isnothing(isigma)
        isigma = sigma \ I
    end
    val = sum(isigma * mu) / sum(isigma)
    return range(; start = val, stop = val, length = length(mu))
end
function target_mean(::MeanSquareError, mu::AbstractArray, sigma::AbstractMatrix;
                     T::Integer, kwargs...)
    val = tr(sigma) / T
    return range(; start = val, stop = val, length = length(mu))
end

"""
    mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}, X::AbstractArray; dims::Int = 1, kwargs...)
    mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}, X::AbstractArray; dims::Int = 1, kwargs...)
    mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}, X::AbstractArray; dims::Int = 1, kwargs...)

Compute shrunk expected returns using the specified estimator.

This method applies a shrinkage algorithm to the sample expected returns, pulling them toward a specified target to reduce estimation error, especially in high-dimensional settings.

# Arguments

  - `me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}`: Use the James-Stein algorithm.
  - `me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}`: Use the Bayes-Stein algorithm.
  - `me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}`: Use the Bodnar-Okhrin-Parolya algorithm.
  - `X::AbstractArray`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the mean.
  - `kwargs...`: Additional keyword arguments passed to the mean and covariance estimators.

# Returns

  - `mu::AbstractArray`: Shrunk expected returns vector.

# Details

  - Computes the sample mean and covariance.

  - Computes the shrinkage target using `target_mean`.
  - Computes the shrinkage intensity `alpha` with:

      + `JamesStein`: the centered mean and eigenvalues of the covariance matrix.
      + `BayesStein`: a Bayesian formula involving the centered mean and inverse covariance.
      + `BodnarOkhrinParolya`: a Bayesian formula involving the target mean, mean and inverse covariance.
  - Returns the shrunk mean vector.

# Related

  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein},
                         X::AbstractArray; dims::Int = 1, kwargs...)
    mu = mean(me.me, X; dims = dims, kwargs...)
    sigma = cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    b = if isone(dims)
        transpose(target_mean(me.alg.target, transpose(mu), sigma; T = T))
    else
        target_mean(me.alg.target, mu, sigma; T = T)
    end
    evals = eigvals(sigma)
    mb = mu - b
    alpha = (N * mean(evals) - 2 * maximum(evals)) / dot(mb, mb) / T
    return (one(alpha) - alpha) * mu + alpha * b
end
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein},
                         X::AbstractArray; dims::Int = 1, kwargs...)
    mu = mean(me.me, X; dims = dims, kwargs...)
    sigma = cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    isigma = sigma \ I
    b = if isone(dims)
        transpose(target_mean(me.alg.target, transpose(mu), sigma; isigma = isigma, T = T))
    else
        target_mean(me.alg.target, mu, sigma; isigma = isigma, T = T)
    end
    mb = vec(mu - b)
    alpha = (N + 2) / ((N + 2) + T * dot(mb, isigma, mb))
    return (one(alpha) - alpha) * mu + alpha * b
end
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya},
                         X::AbstractArray; dims::Int = 1, kwargs...)
    mu = mean(me.me, X; dims = dims, kwargs...)
    sigma = cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    isigma = sigma \ I
    b = if isone(dims)
        transpose(target_mean(me.alg.target, transpose(mu), sigma; isigma = isigma, T = T))
    else
        target_mean(me.alg.target, mu, sigma; isigma = isigma, T = T)
    end
    u = dot(reshape(mu, :, 1), isigma, reshape(mu, :, 1))
    v = dot(reshape(b, :, 1), isigma, reshape(b, :, 1))
    w = dot(reshape(mu, :, 1), isigma, reshape(b, :, 1))
    alpha = (u - N / (T - N)) * v - w^2
    alpha /= u * v - w^2
    beta = (one(alpha) - alpha) * w / u
    return alpha * mu + beta * b
end
function factory(ce::ShrunkExpectedReturns, w::Union{Nothing, <:AbstractWeights} = nothing)
    return ShrunkExpectedReturns(; me = factory(ce.me, w), ce = factory(ce.ce, w),
                                 alg = ce.alg)
end

export GrandMean, VolatilityWeighted, MeanSquareError, JamesStein, BayesStein,
       BodnarOkhrinParolya, ShrunkExpectedReturns
