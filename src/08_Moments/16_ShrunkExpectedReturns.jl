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
  - [`MeanSquaredError`](@ref)
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
    struct MeanSquaredError <: AbstractShrunkExpectedReturnsTarget end

Shrinkage target representing the mean squared error of expected returns.

`MeanSquaredError` computes the shrinkage target as the trace of the covariance matrix divided by the number of observations, resulting in a vector where each element is the same value. This target is useful for certain shrinkage estimators that minimize mean squared error.

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
struct MeanSquaredError <: AbstractShrunkExpectedReturnsTarget end
"""
    struct JamesStein{T1} <: AbstractShrunkExpectedReturnsAlgorithm
        tgt::T1
    end

Shrinkage algorithm implementing the James-Stein estimator for expected returns.

`JamesStein` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean). The estimator reduces estimation error, especially in high-dimensional settings.

# Fields

  - `tgt`: The shrinkage target type.

# Constructor

    JamesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> JamesStein()
JamesStein
  tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
"""
struct JamesStein{T1} <: AbstractShrunkExpectedReturnsAlgorithm
    tgt::T1
    function JamesStein(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function JamesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())
    return JamesStein(tgt)
end
"""
    struct BayesStein{T1} <: AbstractShrunkExpectedReturnsAlgorithm
        tgt::T1
    end

Shrinkage algorithm implementing the Bayes-Stein estimator for expected returns.

`BayesStein` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean) using Bayesian principles. This estimator is useful for reducing estimation error, especially when sample sizes are small.

# Fields

  - `tgt`: The shrinkage target type.

# Constructor

    BayesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> BayesStein()
BayesStein
  tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
"""
struct BayesStein{T1} <: AbstractShrunkExpectedReturnsAlgorithm
    tgt::T1
    function BayesStein(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function BayesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())
    return BayesStein(tgt)
end
"""
    struct BodnarOkhrinParolya{T1} <: AbstractShrunkExpectedReturnsAlgorithm
        tgt::T1
    end

Shrinkage algorithm implementing the Bodnar-Okhrin-Parolya estimator for expected returns.

`BodnarOkhrinParolya` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean) using the Bodnar-Okhrin-Parolya approach. This estimator is designed for robust estimation in high-dimensional settings.

# Fields

  - `tgt`: The shrinkage target type.

# Constructor

    BodnarOkhrinParolya(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> BodnarOkhrinParolya()
BodnarOkhrinParolya
  tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
"""
struct BodnarOkhrinParolya{T1} <: AbstractShrunkExpectedReturnsAlgorithm
    tgt::T1
    function BodnarOkhrinParolya(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function BodnarOkhrinParolya(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())
    return BodnarOkhrinParolya(tgt)
end
"""
    struct ShrunkExpectedReturns{T1, T2, T3} <: AbstractShrunkExpectedReturnsEstimator
        me::T1
        ce::T2
        alg::T3
    end

Container type for shrinkage-based expected returns estimators.

`ShrunkExpectedReturns` encapsulates all components required for shrinkage estimation of expected returns, including the mean estimator, covariance estimator, and shrinkage algorithm. This enables modular and extensible workflows for robust expected returns estimation using shrinkage techniques.

# Fields

  - `me`: Mean estimator for expected returns.
  - `ce`: Covariance estimator.
  - `alg`: Shrinkage algorithm (e.g., James-Stein, Bayes-Stein).

# Constructor

    ShrunkExpectedReturns(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                          alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> ShrunkExpectedReturns()
ShrunkExpectedReturns
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
   ce ┼ PortfolioOptimisersCovariance
      │   ce ┼ Covariance
      │      │    me ┼ SimpleExpectedReturns
      │      │       │   w ┴ nothing
      │      │    ce ┼ GeneralCovariance
      │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │      │       │    w ┴ nothing
      │      │   alg ┴ Full()
      │   mp ┼ DenoiseDetoneAlgMatrixProcessing
      │      │       pdm ┼ Posdef
      │      │           │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │      │           │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      │   denoise ┼ nothing
      │      │    detone ┼ nothing
      │      │       alg ┼ nothing
      │      │     order ┴ DenoiseDetoneAlg()
  alg ┼ JamesStein
      │   tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
"""
struct ShrunkExpectedReturns{T1, T2, T3} <: AbstractShrunkExpectedReturnsEstimator
    me::T1
    ce::T2
    alg::T3
    function ShrunkExpectedReturns(me::AbstractExpectedReturnsEstimator,
                                   ce::StatsBase.CovarianceEstimator,
                                   alg::AbstractShrunkExpectedReturnsAlgorithm)
        return new{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
    end
end
function ShrunkExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein())
    return ShrunkExpectedReturns(me, ce, alg)
end
"""
    target_mean(::AbstractShrunkExpectedReturnsTarget, mu::ArrNum, sigma::MatNum;
                kwargs...)

Compute the shrinkage target vector for expected returns estimation.

`target_mean` computes the target vector toward which expected returns are shrunk, based on the specified shrinkage target type. This function is used internally by shrinkage estimators such as James-Stein, Bayes-Stein, and Bodnar-Okhrin-Parolya.

# Arguments

  - `tgt`: The shrinkage target type.

      + `tgt::GrandMean`: Returns a vector filled with the mean of `mu`.
      + `tgt::VolatilityWeighted`: Returns a vector filled with the volatility-weighted mean of `mu`, using the inverse covariance matrix.
      + `tgt::MeanSquaredError`: Returns a vector filled with the trace of `sigma` divided by `T`.

  - `mu`: 1D array of expected returns.
  - `sigma`: Covariance matrix of asset returns.
  - `kwargs...`: Additional keyword arguments, such as `T` (number of observations) or `isigma` (inverse covariance matrix).

# Returns

  - `b::ArrNum`: Target vector for shrinkage estimation.

# Related

  - [`GrandMean`](@ref)
  - [`VolatilityWeighted`](@ref)
  - [`MeanSquaredError`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
function target_mean(::GrandMean, mu::ArrNum, sigma::MatNum; kwargs...)
    val = mean(mu)
    return range(val, val; length = length(mu))
end
function target_mean(::VolatilityWeighted, mu::ArrNum, sigma::MatNum; isigma = nothing,
                     kwargs...)
    if isnothing(isigma)
        isigma = sigma \ LinearAlgebra.I
    end
    val = sum(isigma * mu) / sum(isigma)
    return range(val, val; length = length(mu))
end
function target_mean(::MeanSquaredError, mu::ArrNum, sigma::MatNum; T::Integer, kwargs...)
    val = LinearAlgebra.tr(sigma) / T
    return range(val, val; length = length(mu))
end
"""
    mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute shrunk expected returns using the specified estimator.

This method applies a shrinkage algorithm to the sample expected returns, pulling them toward a specified target to reduce estimation error, especially in high-dimensional settings.

# Arguments

  - `me`: Shrunk expected returns estimator.

      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}`: Use the James-Stein algorithm.
      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}`: Use the Bayes-Stein algorithm.
      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}`: Use the Bodnar-Okhrin-Parolya algorithm.

  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the mean.
  - `kwargs...`: Additional keyword arguments passed to the mean and covariance estimators.

# Returns

  - `mu::ArrNum`: Shrunk expected returns vector.

# Details

  - Computes the sample mean and covariance.

  - Computes the shrinkage target using `target_mean`.
  - Computes the shrinkage intensity `alpha` with:

      + `JamesStein`: The centered mean and eigenvalues of the covariance matrix.
      + `BayesStein`: A Bayesian formula involving the centered mean and inverse covariance.
      + `BodnarOkhrinParolya`: A Bayesian formula involving the target mean, mean and inverse covariance.
  - ReturnsResult the shrunk mean vector.

# Related

  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}, X::MatNum;
                         dims::Int = 1, kwargs...)
    mu = mean(me.me, X; dims = dims, kwargs...)
    sigma = cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    b = if isone(dims)
        transpose(target_mean(me.alg.tgt, transpose(mu), sigma; T = T))
    else
        target_mean(me.alg.tgt, mu, sigma; T = T)
    end
    evals = eigvals(sigma)
    mb = mu - b
    alpha = (N * mean(evals) - 2 * maximum(evals)) / LinearAlgebra.dot(mb, mb) / T
    return (one(alpha) - alpha) * mu + alpha * b
end
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}, X::MatNum;
                         dims::Int = 1, kwargs...)
    mu = mean(me.me, X; dims = dims, kwargs...)
    sigma = cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    isigma = sigma \ LinearAlgebra.I
    b = if isone(dims)
        transpose(target_mean(me.alg.tgt, transpose(mu), sigma; isigma = isigma, T = T))
    else
        target_mean(me.alg.tgt, mu, sigma; isigma = isigma, T = T)
    end
    mb = vec(mu - b)
    alpha = (N + 2) / ((N + 2) + T * LinearAlgebra.dot(mb, isigma, mb))
    return (one(alpha) - alpha) * mu + alpha * b
end
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya},
                         X::MatNum; dims::Int = 1, kwargs...)
    mu = mean(me.me, X; dims = dims, kwargs...)
    sigma = cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    isigma = sigma \ LinearAlgebra.I
    b = if isone(dims)
        transpose(target_mean(me.alg.tgt, transpose(mu), sigma; isigma = isigma, T = T))
    else
        target_mean(me.alg.tgt, mu, sigma; isigma = isigma, T = T)
    end
    u = LinearAlgebra.dot(reshape(mu, :, 1), isigma, reshape(mu, :, 1))
    v = LinearAlgebra.dot(reshape(b, :, 1), isigma, reshape(b, :, 1))
    w = LinearAlgebra.dot(reshape(mu, :, 1), isigma, reshape(b, :, 1))
    alpha = (u - N / (T - N)) * v - w^2
    alpha /= u * v - w^2
    beta = (one(alpha) - alpha) * w / u
    return alpha * mu + beta * b
end
function factory(ce::ShrunkExpectedReturns, w::Option{<:AbstractWeights} = nothing)
    return ShrunkExpectedReturns(; me = factory(ce.me, w), ce = factory(ce.ce, w),
                                 alg = ce.alg)
end

export GrandMean, VolatilityWeighted, MeanSquaredError, JamesStein, BayesStein,
       BodnarOkhrinParolya, ShrunkExpectedReturns
