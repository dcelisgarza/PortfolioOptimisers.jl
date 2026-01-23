"""
    factory(ce::StatsBase.CovarianceEstimator, args...)

Fallback for covariance estimator factory methods.

# Arguments

  - $(glossary[:ce])
  - `args...`: Optional arguments (ignored for base covariance estimators).

# Returns

  - `ce::StatsBase.CovarianceEstimator`: The original covariance estimator.

# Related

  - [`factory`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function factory(ce::StatsBase.CovarianceEstimator, args...)
    return ce
end
"""
    abstract type AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator end

Abstract supertype for all covariance estimator types in PortfolioOptimisers.jl.

All concrete types that implement covariance estimation should subtype `AbstractCovarianceEstimator`.

# Related

  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
  - [`AbstractMomentAlgorithm`](@ref)
"""
abstract type AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator end
"""
    abstract type AbstractVarianceEstimator <: AbstractCovarianceEstimator end

Abstract supertype for all variance estimator types in PortfolioOptimisers.jl.

All concrete types that implement variance estimation should subtype `AbstractVarianceEstimator`.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type AbstractVarianceEstimator <: AbstractCovarianceEstimator end
@define_pretty_show(AbstractCovarianceEstimator)
"""
    abstract type AbstractExpectedReturnsEstimator <: AbstractEstimator end

Abstract supertype for all expected returns estimator types in PortfolioOptimisers.jl.

All concrete types that implement expected returns estimation should subtype `AbstractExpectedReturnsEstimator`.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractExpectedReturnsAlgorithm`](@ref)
"""
abstract type AbstractExpectedReturnsEstimator <: AbstractEstimator end
"""
    abstract type AbstractExpectedReturnsAlgorithm <: AbstractAlgorithm end

Abstract supertype for all expected returns algorithm types in PortfolioOptimisers.jl.

All concrete types that implement a specific algorithm for expected returns estimation (e.g., shrinkage, robust mean) should subtype `AbstractExpectedReturnsAlgorithm`. This allows for flexible extension and dispatch of expected returns estimation routines.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
abstract type AbstractExpectedReturnsAlgorithm <: AbstractAlgorithm end
"""
    abstract type AbstractMomentAlgorithm <: AbstractAlgorithm end

Abstract supertype for all moment algorithm types in PortfolioOptimisers.jl.

All concrete types that implement a specific algorithm for moment estimation (e.g., full, semi) should subtype `AbstractMomentAlgorithm`. This allows for flexible extension and dispatch of moment estimation routines.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type AbstractMomentAlgorithm <: AbstractAlgorithm end
"""
    struct Full <: AbstractMomentAlgorithm end

`Full` is used to indicate that all available data points are included in the moment estimation process.

# Related

  - [`AbstractMomentAlgorithm`](@ref)
  - [`Semi`](@ref)
"""
struct Full <: AbstractMomentAlgorithm end
"""
    struct Semi <: AbstractMomentAlgorithm end

`Semi` is used for semi-moment estimators, where only observations below the mean (i.e., negative deviations) are considered.

# Related

  - [`AbstractMomentAlgorithm`](@ref)
  - [`Full`](@ref)
"""
struct Semi <: AbstractMomentAlgorithm end
"""
    robust_cov(ce::StatsBase.CovarianceEstimator, X::MatNum, [w::StatsBase.AbstractWeights];
               dims::Int = 1, mean = nothing, kwargs...)

Compute the covariance matrix robustly using the specified covariance estimator `ce`, data matrix `X`, and optional weights vector `w`.

This function attempts to compute the weighted covariance matrix using the provided estimator and keyword arguments. If an error occurs (e.g., due to unsupported keyword arguments), it retries with a reduced set of arguments for compatibility. This ensures robust weighted covariance estimation across different estimator types and StatsBase versions.

# Arguments

  - `ce`: Covariance estimator to use.
  - `X`: Data matrix.
  - `w`: Optional weights for each observation.
  - `dims`: Dimension along which to compute the covariance.
  - `mean`: Optional mean array to use for centering.
  - `kwargs...`: Additional keyword arguments passed to `cov`.

# Returns

  - `sigma::MatNum`: Covariance matrix.

# Related

  - [`MatNum`](@ref)
  - [`robust_cor`](@ref)
  - [`Statistics.cov`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function robust_cov(ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    return try
        Statistics.cov(ce, X; dims = dims, mean = mean, kwargs...)
    catch
        Statistics.cov(ce, X; dims = dims, mean = mean)
    end
    #=
    return if hasmethod(cov, (typeof(ce), typeof(X)), (:dims, :mean, :my_kwargs))
        Statistics.cov(ce, X; dims = dims, mean = mean, kwargs...)
    elseif hasmethod(cov, (typeof(ce), typeof(X)), (:dims, :mean))
        Statistics.cov(ce, X; dims = dims, mean = mean)
    end
    =#
end
function robust_cov(ce::StatsBase.CovarianceEstimator, X::MatNum,
                    w::StatsBase.AbstractWeights; dims::Int = 1, mean = nothing, kwargs...)
    return try
        Statistics.cov(ce, X, w; dims = dims, mean = mean, kwargs...)
    catch
        Statistics.cov(ce, X, w; dims = dims, mean = mean)
    end
    #=
    return if hasmethod(cov, (typeof(ce), typeof(X), typeof(w)), (:dims, :mean, :my_kwargs))
        Statistics.cov(ce, X, w; dims = dims, mean = mean, kwargs...)
    elseif hasmethod(cov, (typeof(ce), typeof(X), typeof(w)), (:dims, :mean))
        Statistics.cov(ce, X, w; dims = dims, mean = mean)
    end
    =#
end
"""
    robust_cor(ce::StatsBase.CovarianceEstimator, X::MatNum, [w::StatsBase.AbstractWeights];
               dims::Int = 1, mean = nothing, kwargs...)

Compute the correlation matrix robustly using the specified covariance estimator `ce`, data matrix `X`, and optional weights vector `w`.

This function attempts to compute the weighted correlation matrix using the provided estimator and keyword arguments. If an error occurs, it falls back to computing the weighted covariance matrix and then converts it to a correlation matrix. This ensures robust weighted correlation estimation across different estimator types and StatsBase versions.

# Arguments

  - `ce`: Covariance estimator to use.
  - `X`: Data matrix.
  - `w`: Optional weights for each observation.
  - `dims`: Dimension along which to compute the correlation.
  - `mean`: Optional mean array to use for centering.
  - `kwargs...`: Additional keyword arguments passed to `cor`.

# Returns

  - `rho::MatNum`: Correlation matrix.

# Related

  - [`MatNum`](@ref)
  - [`robust_cov`](@ref)
  - [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function robust_cor(ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    return try
        try
            Statistics.cor(ce, X; dims = dims, mean = mean, kwargs...)
        catch
            Statistics.cor(ce, X; dims = dims, mean = mean)
        end
    catch
        sigma = robust_cov(ce, X; dims = dims, mean = mean, kwargs...)
        if ismutable(sigma)
            StatsBase.StatsBase.cov2cor!(sigma, sqrt.(LinearAlgebra.diag(sigma)))
        else
            sigma = StatsBase.StatsBase.cov2cor(Matrix(sigma))
        end
        sigma
    end
    #=
    return if hasmethod(cor, (typeof(ce), typeof(X)), (:dims, :mean, :my_kwargs))
        Statistics.cor(ce, X; dims = dims, mean = mean, kwargs...)
    elseif hasmethod(cor, (typeof(ce), typeof(X)), (:dims, :mean))
        Statistics.cor(ce, X; dims = dims, mean = mean)
    else
        sigma = robust_cov(ce, X; dims = dims, mean = mean, kwargs...)
        if ismutable(sigma)
            StatsBase.StatsBase.cov2cor!(sigma, sqrt.(LinearAlgebra.diag(sigma)))
        else
            sigma = StatsBase.StatsBase.cov2cor(Matrix(sigma))
        end
        sigma
    end
    =#
end
function robust_cor(ce::StatsBase.CovarianceEstimator, X::MatNum,
                    w::StatsBase.AbstractWeights; dims::Int = 1, mean = nothing, kwargs...)
    return try
        try
            Statistics.cor(ce, X, w; dims = dims, mean = mean, kwargs...)
        catch
            Statistics.cor(ce, X, w; dims = dims, mean = mean)
        end
    catch
        sigma = robust_cov(ce, X, w; dims = dims, mean = mean, kwargs...)
        if ismutable(sigma)
            StatsBase.StatsBase.cov2cor!(sigma, sqrt.(LinearAlgebra.diag(sigma)))
        else
            sigma = StatsBase.StatsBase.cov2cor(Matrix(sigma))
        end
        sigma
    end
    #=
    return if hasmethod(cor, (typeof(ce), typeof(X), typeof(w)), (:dims, :mean, :my_kwargs))
        Statistics.cor(ce, X, w; dims = dims, mean = mean, kwargs...)
    elseif hasmethod(cor, (typeof(ce), typeof(X), typeof(w)), (:dims, :mean))
        Statistics.cor(ce, X, w; dims = dims, mean = mean)
    else
        sigma = robust_cov(ce, X, w; dims = dims, mean = mean, kwargs...)
        if ismutable(sigma)
            StatsBase.StatsBase.cov2cor!(sigma, sqrt.(LinearAlgebra.diag(sigma)))
        else
            sigma = StatsBase.StatsBase.cov2cor(Matrix(sigma))
        end
        sigma
    end
    =#
end

export Full, Semi
