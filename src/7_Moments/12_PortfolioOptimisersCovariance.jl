"""
    struct PortfolioOptimisersCovariance{T1 <: AbstractCovarianceEstimator,
                                         T2 <: AbstractMatrixProcessingEstimator} <: AbstractCovarianceEstimator

Composite covariance estimator with post-processing.

`PortfolioOptimisersCovariance` is a flexible container type that combines any covariance estimator with a matrix post-processing step. This enables users to apply additional transformations or corrections (such as shrinkage, regularisation, or projection to positive definite) to the covariance or correlation matrix after it is estimated.

# Fields

  - `ce::AbstractCovarianceEstimator`: The underlying covariance estimator.
  - `mp::AbstractMatrixProcessingEstimator`: Matrix post-processing estimator.

# Constructor

    PortfolioOptimisersCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                                   mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing())

Creates a `PortfolioOptimisersCovariance` object with the specified covariance estimator and matrix processing step.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
"""
struct PortfolioOptimisersCovariance{T1 <: AbstractCovarianceEstimator,
                                     T2 <: AbstractMatrixProcessingEstimator} <:
       AbstractCovarianceEstimator
    ce::T1
    mp::T2
end
"""
    PortfolioOptimisersCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                                   mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing())

Construct a [`PortfolioOptimisersCovariance`](@ref) estimator that applies a matrix post-processing step to the output of any covariance estimator.

This constructor creates a `PortfolioOptimisersCovariance` object using the specified covariance estimator and matrix processing estimator. The resulting object can be used as a drop-in replacement for any covariance estimator, with the added benefit of post-processing (such as shrinkage, regularisation, or projection to positive definite).

# Arguments

  - `ce::AbstractCovarianceEstimator`: Covariance estimator to use.
  - `mp::AbstractMatrixProcessingEstimator`: Matrix post-processing estimator.

# Returns

  - `PortfolioOptimisersCovariance`: A composite covariance estimator with post-processing.

# Examples

```jldoctest
julia> ce = PortfolioOptimisersCovariance()
PortfolioOptimisersCovariance
  ce | Covariance
     |    me | SimpleExpectedReturns
     |       |   w | nothing
     |    ce | GeneralWeightedCovariance
     |       |   ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     |       |    w | nothing
     |   alg | Full()
  mp | DefaultMatrixProcessing
     |       pdm | PosdefEstimator
     |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
     |   denoise | nothing
     |    detone | nothing
     |       alg | nothing
```

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`Covariance`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`DefaultMatrixProcessing`](@ref)
"""
function PortfolioOptimisersCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                                       mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing())
    return PortfolioOptimisersCovariance{typeof(ce), typeof(mp)}(ce, mp)
end
function factory(ce::PortfolioOptimisersCovariance,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return PortfolioOptimisersCovariance(; ce = factory(ce.ce, w), mp = ce.mp)
end

"""
    cov(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1, kwargs...)

Compute the covariance matrix with post-processing using a [`PortfolioOptimisersCovariance`](@ref) estimator.

This method computes the covariance matrix for the input data matrix `X` using the underlying covariance estimator in `ce`, and then applies the matrix post-processing step specified by `ce.mp`. This enables workflows such as shrinkage, regularisation, or projection to positive definite after covariance estimation.

# Arguments

  - `ce::PortfolioOptimisersCovariance`: Composite covariance estimator with post-processing.
  - `X::AbstractMatrix`: Data matrix of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the covariance (1 = columns/assets, 2 = rows). Default is `1`.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and matrix processing step.

# Returns

  - `sigma::Matrix{Float64}`: The processed covariance matrix.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing!`](@ref)
  - [`Statistics.cov`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cov-Tuple%7BCovarianceEstimator,%20AbstractMatrix%7D)
"""
function Statistics.cov(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                        kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = cov(ce.ce, X; kwargs...)
    matrix_processing!(ce.mp, sigma, X; kwargs...)
    return sigma
end

"""
    cor(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1, kwargs...)

Compute the correlation matrix with post-processing using a [`PortfolioOptimisersCovariance`](@ref) estimator.

This method computes the correlation matrix for the input data matrix `X` using the underlying covariance estimator in `ce`, and then applies the matrix post-processing step specified by `ce.mp`. This enables workflows such as shrinkage, regularisation, or projection to positive definite after correlation estimation.

# Arguments

  - `ce::PortfolioOptimisersCovariance`: Composite covariance estimator with post-processing.
  - `X::AbstractMatrix`: Data matrix of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the correlation (1 = columns/assets, 2 = rows). Default is `1`.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and matrix processing step.

# Returns

  - `rho::Matrix{Float64}`: The processed correlation matrix.

# Validation

  - Asserts that `dims` is either `1` or `2`.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing!`](@ref)
  - [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cor)
"""
function Statistics.cor(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                        kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = cor(ce.ce, X; kwargs...)
    matrix_processing!(ce.mp, rho, X; kwargs...)
    return rho
end

export PortfolioOptimisersCovariance
