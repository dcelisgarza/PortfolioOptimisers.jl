"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrunk expected returns estimators in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing shrinkage-based expected returns estimation algorithms should be subtypes of `AbstractShrunkExpectedReturnsEstimator`.

# Related

  - [`ShrunkExpectedReturns`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsEstimator <: AbstractExpectedReturnsEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrinkage algorithms for expected returns estimation.

All concrete and/or abstract types implementing specific shrinkage algorithms (e.g., James-Stein, Bayes-Stein) should be subtypes of `AbstractShrunkExpectedReturnsAlgorithm`.

# Related

  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`AbstractExpectedReturnsAlgorithm`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsAlgorithm <: AbstractExpectedReturnsAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all shrinkage targets used in expected returns estimation.

Concrete types implementing specific shrinkage targets (e.g., grand mean, volatility-weighted mean) should subtype `AbstractShrunkExpectedReturnsTarget`.

# Related

  - [`GrandMean`](@ref)
  - [`VolatilityWeighted`](@ref)
  - [`MeanSquaredError`](@ref)
"""
abstract type AbstractShrunkExpectedReturnsTarget <: AbstractExpectedReturnsAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage target representing the grand mean of expected returns.

`GrandMean` computes the shrinkage target as the mean of all asset expected returns, resulting in a vector where each element is the same grand mean value. This is commonly used in shrinkage estimators to reduce estimation error by pulling individual expected returns toward the overall average.

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
struct GrandMean <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage target representing the volatility-weighted mean of expected returns.

`VolatilityWeighted` computes the shrinkage target as a weighted mean of expected returns, where weights are inversely proportional to asset volatility (from the inverse covariance matrix). This approach accounts for differences in asset risk when estimating the shrinkage target.

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
struct VolatilityWeighted <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage target representing the mean squared error of expected returns.

`MeanSquaredError` computes the shrinkage target as the trace of the covariance matrix divided by the number of observations, resulting in a vector where each element is the same value. This target is useful for certain shrinkage estimators that minimize mean squared error.

# Related

  - [`AbstractShrunkExpectedReturnsTarget`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
"""
struct MeanSquaredError <: AbstractShrunkExpectedReturnsTarget end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage algorithm implementing the James-Stein estimator for expected returns.

`JamesStein` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean). The estimator reduces estimation error, especially in high-dimensional settings.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    JamesStein(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean()
    ) -> JamesStein

Keywords correspond to the struct's fields.

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
@concrete struct JamesStein <: AbstractShrunkExpectedReturnsAlgorithm
    "$(field_dict[:mutgt])"
    tgt
    function JamesStein(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function JamesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())::JamesStein
    return JamesStein(tgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage algorithm implementing the Bayes-Stein estimator for expected returns.

`BayesStein` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean) using Bayesian principles. This estimator is useful for reducing estimation error, especially when sample sizes are small.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BayesStein(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean()
    ) -> BayesStein

Keywords correspond to the struct's fields.

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
@concrete struct BayesStein <: AbstractShrunkExpectedReturnsAlgorithm
    "$(field_dict[:mutgt])"
    tgt
    function BayesStein(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function BayesStein(; tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())::BayesStein
    return BayesStein(tgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Shrinkage algorithm implementing the Bodnar-Okhrin-Parolya estimator for expected returns.

`BodnarOkhrinParolya` applies shrinkage to asset expected returns by pulling them toward a specified target (e.g., grand mean, volatility-weighted mean) using the Bodnar-Okhrin-Parolya approach. This estimator is designed for robust estimation in high-dimensional settings.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BodnarOkhrinParolya(;
        tgt::AbstractShrunkExpectedReturnsTarget = GrandMean()
    ) -> BodnarOkhrinParolya

Keywords correspond to the struct's fields.

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
@concrete struct BodnarOkhrinParolya <: AbstractShrunkExpectedReturnsAlgorithm
    "$(field_dict[:mutgt])"
    tgt
    function BodnarOkhrinParolya(tgt::AbstractShrunkExpectedReturnsTarget)
        return new{typeof(tgt)}(tgt)
    end
end
function BodnarOkhrinParolya(;
                             tgt::AbstractShrunkExpectedReturnsTarget = GrandMean())::BodnarOkhrinParolya
    return BodnarOkhrinParolya(tgt)
end
"""
$(DocStringExtensions.TYPEDEF)

Container type for shrinkage-based expected returns estimators.

`ShrunkExpectedReturns` encapsulates all components required for shrinkage estimation of expected returns, including the mean estimator, covariance estimator, and shrinkage algorithm.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ShrunkExpectedReturns(;
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein()
    ) -> ShrunkExpectedReturns

Keywords correspond to the struct's fields.

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
      │      │     pdm ┼ Posdef
      │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
      │      │      dn ┼ nothing
      │      │      dt ┼ nothing
      │      │     alg ┼ nothing
      │      │   order ┴ DenoiseDetoneAlg()
  alg ┼ JamesStein
      │   tgt ┴ GrandMean()
```

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`AbstractShrunkExpectedReturnsAlgorithm`](@ref)
"""
@concrete struct ShrunkExpectedReturns <: AbstractShrunkExpectedReturnsEstimator
    "$(field_dict[:me])"
    me
    "$(field_dict[:ce])"
    ce
    "$(field_dict[:me_shrink_alg])"
    alg
    function ShrunkExpectedReturns(me::AbstractExpectedReturnsEstimator,
                                   ce::StatsBase.CovarianceEstimator,
                                   alg::AbstractShrunkExpectedReturnsAlgorithm)
        return new{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
    end
end
function ShrunkExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               alg::AbstractShrunkExpectedReturnsAlgorithm = JamesStein())::ShrunkExpectedReturns
    return ShrunkExpectedReturns(me, ce, alg)
end
"""
    target_mean(::AbstractShrunkExpectedReturnsTarget, mu::ArrNum, sigma::MatNum, args...;
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
function target_mean(::GrandMean, mu::ArrNum, sigma::MatNum, args...; kwargs...)
    val = Statistics.mean(mu)
    return range(val, val; length = length(mu))
end
function target_mean(::VolatilityWeighted, mu::ArrNum, sigma::MatNum,
                     isigma::Option{<:MatNum} = nothing; kwargs...)
    if isnothing(isigma)
        isigma = sigma \ LinearAlgebra.I
    end
    if isone(size(mu, 1))
        mu = vec(mu)
    end
    val = sum(isigma * mu) / sum(isigma)
    return range(val, val; length = length(mu))
end
function target_mean(::MeanSquaredError, mu::ArrNum, sigma::MatNum, args...; T::Integer,
                     kwargs...)
    val = LinearAlgebra.tr(sigma) / T
    return range(val, val; length = length(mu))
end
"""
    Statistics.mean(me::ShrunkExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute shrunk expected returns using the specified estimator.

This method applies a shrinkage algorithm to the sample expected returns, pulling them toward a specified target to reduce estimation error, especially in high-dimensional settings.

# Arguments

  - `me`: Shrunk expected returns estimator.

      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}`: Use the James-Stein algorithm.
      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}`: Use the Bayes-Stein algorithm.
      + `me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya}`: Use the Bodnar-Okhrin-Parolya algorithm.

  - `X`: Data matrix (observations × assets).

  - $(arg_dict[:dims])

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

  - Returns the shrunk mean vector.

# Related

  - [`JamesStein`](@ref)
  - [`BayesStein`](@ref)
  - [`BodnarOkhrinParolya`](@ref)
  - [`ShrunkExpectedReturns`](@ref)
  - [`target_mean`](@ref)
"""
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:JamesStein}, X::MatNum;
                         dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    b = target_mean(me.alg.tgt, mu, sigma; T = T)
    if flag
        b = transpose(b)
    end
    evals = LinearAlgebra.eigvals(sigma)
    mb = mu - b
    alpha = (N * Statistics.mean(evals) - 2 * maximum(evals)) / LinearAlgebra.dot(mb, mb) /
            T
    return (one(alpha) - alpha) * mu + alpha * b
end
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BayesStein}, X::MatNum;
                         dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    isigma = sigma \ LinearAlgebra.I
    b = target_mean(me.alg.tgt, mu, sigma, isigma; T = T)
    if flag
        b = transpose(b)
    end
    mb = vec(mu - b)
    alpha = (N + 2) / ((N + 2) + T * LinearAlgebra.dot(mb, isigma, mb))
    return (one(alpha) - alpha) * mu + alpha * b
end
function Statistics.mean(me::ShrunkExpectedReturns{<:Any, <:Any, <:BodnarOkhrinParolya},
                         X::MatNum; dims::Int = 1, kwargs...)
    mu = Statistics.mean(me.me, X; dims = dims, kwargs...)
    sigma = Statistics.cov(me.ce, X; dims = dims, kwargs...)
    T, N = size(X)
    flag = isone(dims)
    if !flag
        N, T = T, N
    end
    isigma = sigma \ LinearAlgebra.I
    b = target_mean(me.alg.tgt, mu, sigma, isigma; T = T)
    if flag
        b = transpose(b)
        vm = vec(mu)
        vb = vec(b)
    else
        vm = mu
        vb = b
    end
    u = LinearAlgebra.dot(vm, isigma, vm)
    v = LinearAlgebra.dot(vb, isigma, vb)
    w = LinearAlgebra.dot(vm, isigma, vb)
    alpha = (u - N / (T - N)) * v - w^2
    alpha /= u * v - w^2
    beta = (one(alpha) - alpha) * w / u
    return alpha * mu + beta * b
end
"""
    factory(ce::ShrunkExpectedReturns, w::ObsWeights) -> ShrunkExpectedReturns

Return a new [`ShrunkExpectedReturns`](@ref) estimator with observation weights `w` applied to the underlying mean and covariance estimators.

# Arguments

  - `me`: Shrunk expected returns estimator.
  - $(arg_dict[:ow])

# Returns

  - `me::ShrunkExpectedReturns`: Updated estimator with weights applied.

# Related

  - [`ShrunkExpectedReturns`](@ref)
  - [`factory`](@ref)
"""
function factory(me::ShrunkExpectedReturns, w::ObsWeights)::ShrunkExpectedReturns
    return ShrunkExpectedReturns(; me = factory(me.me, w), ce = factory(me.ce, w),
                                 alg = me.alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the expected returns estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:me])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:mev])

# Related

  - [`ShrunkExpectedReturns`](@ref)
"""
function moment_view(me::ShrunkExpectedReturns, i)::ShrunkExpectedReturns
    return ShrunkExpectedReturns(; me = moment_view(me.me, i), ce = moment_view(me.ce, i),
                                 alg = me_alg_view(me.alg, i))
end

export GrandMean, VolatilityWeighted, MeanSquaredError, JamesStein, BayesStein,
       BodnarOkhrinParolya, ShrunkExpectedReturns
