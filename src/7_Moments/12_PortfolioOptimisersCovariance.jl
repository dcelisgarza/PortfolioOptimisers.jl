"""
```julia
struct PortfolioOptimisersCovariance{T1, T2} <: AbstractCovarianceEstimator
    ce::T1
    mp::T2
end
```

Composite covariance estimator with post-processing.

`PortfolioOptimisersCovariance` is a flexible container type that combines any covariance estimator with a matrix post-processing step. This enables users to apply additional transformations or corrections (such as shrinkage, regularisation, or projection to positive definite) to the covariance or correlation matrix after it is estimated.

# Fields

  - `ce`: The underlying covariance estimator.
  - `mp`: Matrix post-processing estimator.

# Constructor

    PortfolioOptimisersCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                                    mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing())

Keyword arguments correspond to the fields above.

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
     |       pdm | Posdef
     |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
     |   denoise | nothing
     |    detone | nothing
     |       alg | nothing
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
"""
struct PortfolioOptimisersCovariance{T1, T2} <: AbstractCovarianceEstimator
    ce::T1
    mp::T2
end
function PortfolioOptimisersCovariance(; ce::AbstractCovarianceEstimator = Covariance(),
                                       mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing())
    return PortfolioOptimisersCovariance(ce, mp)
end
function factory(ce::PortfolioOptimisersCovariance,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return PortfolioOptimisersCovariance(; ce = factory(ce.ce, w), mp = ce.mp)
end

"""
```julia
cov(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1, kwargs...)
```

Compute the covariance matrix with post-processing using a [`PortfolioOptimisersCovariance`](@ref) estimator.

This method computes the covariance matrix for the input data matrix `X` using the underlying covariance estimator in `ce`, and then applies the matrix post-processing step specified by `ce.mp`. This enables workflows such as shrinkage, regularisation, or projection to positive definite after covariance estimation.

# Arguments

  - `ce`: Composite covariance estimator with post-processing.
  - `X`: Data matrix of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the covariance (1 = columns/assets, 2 = rows). Default is `1`.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and matrix processing step.

# Returns

  - `sigma::Matrix{<:Real}`: The processed covariance matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing!`](@ref)
  - [`Statistics.cov`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cov-Tuple%7BCovarianceEstimator,%20AbstractMatrix%7D)
"""
function Statistics.cov(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                        kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    sigma = cov(ce.ce, X; kwargs...)
    if !ismutable(sigma)
        sigma = Matrix(sigma)
    end
    matrix_processing!(ce.mp, sigma, X; kwargs...)
    return sigma
end

"""
```julia
cor(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1, kwargs...)
```

Compute the correlation matrix with post-processing using a [`PortfolioOptimisersCovariance`](@ref) estimator.

This method computes the correlation matrix for the input data matrix `X` using the underlying covariance estimator in `ce`, and then applies the matrix post-processing step specified by `ce.mp`. This enables workflows such as shrinkage, regularisation, or projection to positive definite after correlation estimation.

# Arguments

  - `ce`: Composite covariance estimator with post-processing.
  - `X`: Data matrix of asset returns (observations × assets).
  - `dims`: Dimension along which to compute the correlation (1 = columns/assets, 2 = rows). Default is `1`.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator and matrix processing step.

# Returns

  - `rho::Matrix{<:Real}`: The processed correlation matrix.

# Validation

  - `dims` is either `1` or `2`.

# Related

  - [`PortfolioOptimisersCovariance`](@ref)
  - [`matrix_processing!`](@ref)
  - [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cor)
"""
function Statistics.cor(ce::PortfolioOptimisersCovariance, X::AbstractMatrix; dims = 1,
                        kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    rho = cor(ce.ce, X; kwargs...)
    if !ismutable(rho)
        rho = Matrix(rho)
    end
    matrix_processing!(ce.mp, rho, X; kwargs...)
    return rho
end

export PortfolioOptimisersCovariance
