"""
    ProcessedCovariance

A covariance estimator that applies a custom matrix processing algorithm and positive definite projection to the output of another covariance estimator. This type enables flexible post-processing of covariance matrices by first computing a base covariance, then applying a user-specified matrix processing algorithm and positive definiteness correction in sequence.

# Fields

  - `ce`: The underlying covariance estimator to be processed.
  - `alg`: The matrix processing algorithm to apply to the covariance matrix.
  - `pdm`: The positive definite matrix projection method.

# Constructors

```julia
ProcessedCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                    alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                    pdm::Option{<:Posdef} = Posdef())
```

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> ProcessedCovariance()
ProcessedCovariance
   ce ┼ Covariance
      │    me ┼ SimpleExpectedReturns
      │       │   w ┴ nothing
      │    ce ┼ GeneralCovariance
      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      │       │    w ┴ nothing
      │   alg ┴ Full()
  alg ┼ nothing
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`Posdef`](@ref)
"""
struct ProcessedCovariance{T1, T2, T3} <: AbstractCovarianceEstimator
    ce::T1
    alg::T2
    pdm::T3
    function ProcessedCovariance(ce::AbstractCovarianceEstimator,
                                 alg::Option{<:AbstractMatrixProcessingAlgorithm},
                                 pdm::Option{<:Posdef})
        return new{typeof(ce), typeof(alg), typeof(pdm)}(ce, alg, pdm)
    end
end
function ProcessedCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                             alg::Option{<:AbstractMatrixProcessingAlgorithm} = nothing,
                             pdm::Option{<:Posdef} = Posdef())
    return ProcessedCovariance(ce, alg, pdm)
end
function factory(ce::ProcessedCovariance, w::Option{<:StatsBase.AbstractWeights} = nothing)
    return ProcessedCovariance(; ce = factory(ce.ce, w), alg = ce.alg, pdm = ce.pdm)
end
"""
    Statistics.cov(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)

Compute the processed and positive definite projected covariance matrix for the data matrix `X` using the specified `ProcessedCovariance` estimator.

# Arguments

  - `ce`: The `ProcessedCovariance` estimator specifying the base covariance estimator, matrix processing algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - `dims`: The dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and processing algorithm.

# Returns

  - `sigma::MatNum`: processed covariance matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the covariance matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the covariance matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the matrix processing algorithm in `ce.alg` to the covariance matrix.
  - Returns the processed covariance matrix.

# Related

  - [`ProcessedCovariance`](@ref)
  - [`cor(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cov(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = Statistics.cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    posdef!(ce.pdm, sigma)
    matrix_processing_algorithm!(ce.alg, sigma, X; kwargs...)
    return sigma
end
"""
    Statistics.cor(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)

Compute the processed and positive definite projected correlation matrix for the data matrix `X` using the specified `ProcessedCovariance` estimator.

# Arguments

  - `ce`: The `ProcessedCovariance` estimator specifying the base covariance estimator, matrix processing algorithm, and positive definite projection.
  - `X`: The data matrix (observations × assets).
  - `dims`: The dimension along which to compute the correlation.
  - `kwargs...`: Additional keyword arguments passed to the underlying correlation estimator and processing algorithm.

# Returns

  - `rho::MatNum`: processed correlation matrix.

# Validation

  - `dims in (1, 2)`.

# Details

  - Computes the correlation matrix using the base estimator in `ce`.
  - Transposes `X` if `dims == 2` to ensure variables are in columns.
  - Ensures the correlation matrix is mutable.
  - Applies positive definite projection using the method in `ce.pdm`.
  - Applies the matrix processing algorithm in `ce.alg` to the correlation matrix.
  - Returns the processed correlation matrix.

# Related

  - [`ProcessedCovariance`](@ref)
  - [`cov(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`Posdef`](@ref)
"""
function Statistics.cor(ce::ProcessedCovariance, X::MatNum; dims = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = Statistics.cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    posdef!(ce.pdm, rho)
    matrix_processing_algorithm!(ce.alg, rho, X; kwargs...)
    return rho
end

export ProcessedCovariance
