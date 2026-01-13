"""
    struct DetoneCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
        ce::T1
        dt::T2
        pdm::T3
    end

A covariance estimator that applies a detoning algorithm and positive definite projection to the output of another covariance estimator. This type enables robust estimation of covariance matrices by first computing a base covariance, then applying detoning and positive definiteness corrections in sequence.

# Fields

  - `ce`: The underlying covariance estimator to be detoned.
  - `dt`: The detoning algorithm to apply to the covariance matrix.
  - `pdm`: The positive definite matrix projection method.

# Constructors

```julia
DetoneCovariance(; ce::AbstractCovarianceEstimator = Covariance(), dt::Detone = Detone(),
                 pdm::Option{<:Posdef} = Posdef())
```

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> DetoneCovariance()
DetoneCovariance
      ce ┼ Covariance
         │    me ┼ SimpleExpectedReturns
         │       │   w ┴ nothing
         │    ce ┼ GeneralCovariance
         │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
         │       │    w ┴ nothing
         │   alg ┴ Full()
  detone ┼ Detone
         │     n ┼ Int64: 1
         │   pdm ┼ Posdef
         │       │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │       │   kwargs ┴ @NamedTuple{}: NamedTuple()
     pdm ┼ Posdef
         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
         │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`Detone`](@ref)
  - [`Posdef`](@ref)
"""
struct DetoneCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
    ce::T1
    dt::T2
    pdm::T3
    function DetoneCovariance(ce::AbstractCovarianceEstimator, dt::Detone,
                              pdm::Option{<:Posdef})
        return new{typeof(ce), typeof(dt), typeof(pdm)}(ce, dt, pdm)
    end
end
function DetoneCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                          dt::Detone = Detone(), pdm::Option{<:Posdef} = Posdef())
    return DetoneCovariance(ce, dt, pdm)
end
function factory(ce::DetoneCovariance, w::StatsBase.AbstractWeights)
    return DetoneCovariance(; ce = factory(ce.ce, w), dt = ce.dt, pdm = ce.pdm)
end
"""
    Statistics.cov(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)

Compute the detoned and positive definite projected covariance matrix for the data matrix `X` using the specified `DetoneCovariance` estimator.

# Arguments

  - `ce`: The `DetoneCovariance` estimator specifying the base covariance estimator, detoning algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - `dims`: The dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

# Returns

  - `sigma::MatNum`: detoned covariance matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the covariance matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the covariance matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the detoning algorithm in `ce.dt`.
  - Returns the processed covariance matrix.

# Related

  - [`DetoneCovariance`](@ref)
  - [`cor(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`Detone`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cov(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = Statistics.cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    posdef!(ce.pdm, sigma)
    detone!(ce.dt, sigma)
    return sigma
end
"""
    Statistics.cor(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)

Compute the detoned and positive definite projected correlation matrix for the data matrix `X` using the specified `DetoneCovariance` estimator.

# Arguments

  - `ce`: The `DetoneCovariance` estimator specifying the base covariance estimator, detoning algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - `dims`: The dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the underlying correlation estimator.

# Returns

  - `rho::MatNum`: detoned correlation matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the correlation matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the correlation matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the detoning algorithm in `ce.dt`.
  - Returns the processed correlation matrix.

# Related

  - [`DetoneCovariance`](@ref)
  - [`cov(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`Detone`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cor(ce::DetoneCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = Statistics.cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    posdef!(ce.pdm, rho)
    detone!(ce.dt, rho)
    return rho
end

export DetoneCovariance
