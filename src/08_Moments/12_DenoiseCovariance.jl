"""
    struct DenoiseCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
        ce::T1
        denoise::T2
        pdm::T3
        function DenoiseCovariance(ce::AbstractCovarianceEstimator, denoise::Denoise,
                                   pdm::Option{<:Posdef})
            return new{typeof(ce), typeof(denoise), typeof(pdm)}(ce, denoise, pdm)
        end
    end

A covariance estimator that applies a denoising algorithm and positive definite projection to the output of another covariance estimator. This type enables robust estimation of covariance matrices by first computing a base covariance, then applying denoising and positive definiteness corrections in sequence.

# Fields

  - `ce`: The underlying covariance estimator to be denoised.
  - `denoise`: The denoising algorithm to apply to the covariance matrix.
  - `pdm`: The positive definite matrix projection method.

# Constructors

```julia
DenoiseCovariance(; ce::AbstractCovarianceEstimator; denoise::Denoise = Denoise(),
                  pdm::Option{<:Posdef} = Posdef())
```

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> DenoiseCovariance()
DenoiseCovariance
       ce ┼ Covariance
          │    me ┼ SimpleExpectedReturns
          │       │   w ┴ nothing
          │    ce ┼ GeneralCovariance
          │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
          │       │    w ┴ nothing
          │   alg ┴ Full()
  denoise ┼ Denoise
          │      alg ┼ ShrunkDenoise
          │          │   alpha ┴ Float64: 0.0
          │     args ┼ Tuple{}: ()
          │   kwargs ┼ @NamedTuple{}: NamedTuple()
          │   kernel ┼ typeof(AverageShiftedHistograms.Kernels.gaussian): AverageShiftedHistograms.Kernels.gaussian
          │        m ┼ Int64: 10
          │        n ┼ Int64: 1000
          │      pdm ┼ Posdef
          │          │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
          │          │   kwargs ┴ @NamedTuple{}: NamedTuple()
      pdm ┼ Posdef
          │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
          │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`Denoise`](@ref)
  - [`Posdef`](@ref)
"""
struct DenoiseCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
    ce::T1
    denoise::T2
    pdm::T3
    function DenoiseCovariance(ce::AbstractCovarianceEstimator, denoise::Denoise,
                               pdm::Option{<:Posdef})
        return new{typeof(ce), typeof(denoise), typeof(pdm)}(ce, denoise, pdm)
    end
end
function DenoiseCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                           denoise::Denoise = Denoise(), pdm::Option{<:Posdef} = Posdef())
    return DenoiseCovariance(ce, denoise, pdm)
end
function factory(ce::DenoiseCovariance, w::Option{<:AbstractWeights} = nothing)
    return DenoiseCovariance(; ce = factory(ce.ce, w), denoise = ce.denoise, pdm = ce.pdm)
end
"""
    cov(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)

Compute the denoised and positive definite projected covariance matrix for the data matrix `X` using the specified `DenoiseCovariance` estimator.

# Arguments

  - `ce`: The `DenoiseCovariance` estimator specifying the base covariance estimator, denoising algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - `dims`: The dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

# Returns

  - `sigma::MatNum`: denoised covariance matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the covariance matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the covariance matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the denoising algorithm in `ce.denoise` with the aspect ratio `T/N`.
  - Returns the processed covariance matrix.

# Related

  - [`DenoiseCovariance`](@ref)
  - [`cor(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`Denoise`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cov(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    T, N = size(X)
    posdef!(ce.pdm, sigma)
    denoise!(ce.denoise, sigma, T / N)
    return sigma
end
"""
    cor(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)

Compute the denoised and positive definite projected correlation matrix for the data matrix `X` using the specified `DenoiseCovariance` estimator.

# Arguments

  - `ce`: The `DenoiseCovariance` estimator specifying the base covariance estimator, denoising algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - `dims`: The dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the underlying correlation estimator.

# Returns

  - `rho::MatNum`: denoised correlation matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the correlation matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the correlation matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the denoising algorithm in `ce.denoise` with the aspect ratio `T/N`.
  - Returns the processed correlation matrix.

# Related

  - [`DenoiseCovariance`](@ref)
  - [`cov(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`Denoise`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cor(ce::DenoiseCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    T, N = size(X)
    posdef!(ce.pdm, rho)
    denoise!(ce.denoise, rho, T / N)
    return rho
end

export DenoiseCovariance
