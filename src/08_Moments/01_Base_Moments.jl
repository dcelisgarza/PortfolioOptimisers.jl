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

# Interfaces

In order to implement a new covariance estimator which will work seamlessly with the library, subtype `AbstractCovarianceEstimator` with all necessary parameters---including observation weights---as part of the struct, and implement the following methods:

## Covariance and correlation

  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum; kwargs...)`]: Covariance matrix estimation.
  - [`Statistics.cor(ce::AbstractCovarianceEstimator, X::MatNum; kwargs...)`]: Correlation matrix estimation.

### Arguments

  - $(glossary[:ce])
  - $(glossary[:X])
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

### Returns

  - `sigma::MatNum`: Covariance matrix.

## Factory method

  - [`factory(ce::AbstractCovarianceEstimator, w::StatsBase.AbstractWeights)`]: Factory method for creating instances of the estimator.

### Arguments

  - $(glossary[:ce])
  - $(glossary[:ow])

### Returns

  - $(glossary[:nce])

# Examples

We can create a dummy covariance estimator as follows:

```jldoctest
julia> struct MyCovarianceEstimator{T1} <: PortfolioOptimisers.AbstractCovarianceEstimator
           w::T1
           function MyCovarianceEstimator(w::PortfolioOptimisers.Option{<:StatsBase.AbstractWeights})
               if !isnothing(w) && isempty(w)
                   throw(IsEmptyError("`w` cannot be an empty weights object"))
               end
               return new{typeof(w)}(w)
           end
       end

julia> function MyCovarianceEstimator(;
                                      w::PortfolioOptimisers.Option{<:StatsBase.AbstractWeights} = nothing)
           return MyCovarianceEstimator(w)
       end

julia> function factory(::MyCovarianceEstimator, w::StatsBase.AbstractWeights)
           return MyCovarianceEstimator(; w = w)
       end

julia> function Statistics.cov(est::MyCovarianceEstimator, X::PortfolioOptimisers.MatNum;
                               dims::Int = 1, kwargs...)
           if !(dims in (1, 2))
               throw(DomainError(dims, "dims must be either 1 or 2"))
           end
           if dims == 2
               X = X'
           end
           w = ifelse(isnothing(est.w), StatsBase.fweights(fill(1.0, size(X, 1))), est.w)
           X = X .* w
           sigma = X * X'
           return sigma
       end

julia> function Statistics.cor(est::MyCovarianceEstimator, X::PortfolioOptimisers.MatNum;
                               dims::Int = 1, kwargs...)
           if !(dims in (1, 2))
               throw(DomainError(dims, "dims must be either 1 or 2"))
           end
           if dims == 2
               X = X'
           end
           w = ifelse(isnothing(est.w), StatsBase.fweights(fill(1.0, size(X, 1))), est.w)
           X = X .* w
           sigma = X * X'
           d = LinearAlgebra.diag(sigma)
           StatsBase.cov2cor!(sigma, sqrt.(d))
           return sigma
       end

julia> cov(MyCovarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
3×3 Matrix{Float64}:
 5.0  1.7   2.7
 1.7  0.58  0.92
 2.7  0.92  1.46

julia> cor(MyCovarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
3×3 Matrix{Float64}:
 1.0       0.998274  0.999315
 0.998274  1.0       0.999764
 0.999315  0.999764  1.0
```

# Related

  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
  - [`AbstractMomentAlgorithm`](@ref)
"""
abstract type AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator end
"""
    abstract type AbstractVarianceEstimator <: AbstractCovarianceEstimator end

Abstract supertype for all variance estimator types in PortfolioOptimisers.jl.

All concrete types that implement variance estimation should subtype `AbstractVarianceEstimator`.

# Interfaces

In order to implement a new covariance estimator which will work seamlessly with the library, subtype `AbstractVarianceEstimator` with all necessary parameters---including observation weights---as part of the struct, and implement the following methods:

## Variance and standard deviation

  - `Statistics.var(ve::AbstractVarianceEstimator, X::MatNum; kwargs...)`: Variance estimation.
  - `Statistics.std(ve::AbstractVarianceEstimator, X::MatNum; kwargs...)`: Standard deviation estimation.
  - `Statistics.var(ve::AbstractVarianceEstimator, X::VecNum; kwargs...)`: Variance estimation.
  - `Statistics.std(ve::AbstractVarianceEstimator, X::VecNum; kwargs...)`: Standard deviation estimation.

### Arguments

  - $(glossary[:ve])

  - `X`

      + $(glossary[:X])
      + $(glossary[:Xv])
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

### Returns

  - $(glossary[:X])

      + `val::MatNum`: Variance or standard deviation vector of `X`, reshaped to be consistent with the dimension along which the value is computed.

  - $(glossary[:Xv])

      + `val::VecNum`: Variance or standard deviation of `X`.

## Factory method

  - `factory(ve::AbstractVarianceEstimator, w::StatsBase.AbstractWeights)`: Factory method for creating instances of the estimator.

### Arguments

  - $(glossary[:ve])
  - $(glossary[:ow])

### Returns

  - $(glossary[:nve])

# Examples

We can create a dummy variance estimator as follows:

```jldoctest
julia> struct MyVarianceEstimator{T1} <: PortfolioOptimisers.AbstractVarianceEstimator
           w::T1
           function MyVarianceEstimator(w::PortfolioOptimisers.Option{<:StatsBase.AbstractWeights})
               if !isnothing(w) && isempty(w)
                   throw(IsEmptyError("`w` cannot be an empty weights object"))
               end
               return new{typeof(w)}(w)
           end
       end

julia> function MyVarianceEstimator(;
                                    w::PortfolioOptimisers.Option{<:StatsBase.AbstractWeights} = nothing)
           return MyVarianceEstimator(w)
       end
MyVarianceEstimator

julia> function factory(::MyVarianceEstimator, w::StatsBase.AbstractWeights)
           return MyVarianceEstimator(; w = w)
       end
factory (generic function with 1 method)

julia> function Statistics.var(est::MyVarianceEstimator, X::PortfolioOptimisers.MatNum;
                               dims::Int = 1, kwargs...)
           if !(dims in (1, 2))
               throw(DomainError(dims, "dims must be either 1 or 2"))
           end
           if dims == 2
               X = X'
           end
           w = ifelse(isnothing(est.w), StatsBase.fweights(fill(1.0, size(X, 1))), est.w)
           X = X .* w
           sigma = diag(X * X')
           return isone(dims) ? reshape(sigma, :, 1) : reshape(sigma, 1, :)
       end

julia> function Statistics.std(est::MyVarianceEstimator, X::PortfolioOptimisers.MatNum;
                               dims::Int = 1, kwargs...)
           if !(dims in (1, 2))
               throw(DomainError(dims, "dims must be either 1 or 2"))
           end
           if dims == 2
               X = X'
           end
           w = ifelse(isnothing(est.w), StatsBase.fweights(fill(1.0, size(X, 1))), est.w)
           X = X .* w
           sigma = sqrt.(diag(X * X'))
           return isone(dims) ? reshape(sigma, :, 1) : reshape(sigma, 1, :)
       end

julia> function Statistics.var(est::MyVarianceEstimator, X::PortfolioOptimisers.VecNum; kwargs...)
           w = ifelse(isnothing(est.w), StatsBase.fweights(fill(1.0, size(X, 1))), est.w)
           X = X .* w
           return mean(diag(X' * X))
       end

julia> function Statistics.std(est::MyVarianceEstimator, X::PortfolioOptimisers.VecNum; kwargs...)
           w = ifelse(isnothing(est.w), StatsBase.fweights(fill(1.0, size(X, 1))), est.w)
           X = X .* w
           return sqrt(mean(diag(X' * X)))
       end

julia> var(MyVarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
3×1 Matrix{Float64}:
 5.0
 0.58
 1.4600000000000002

julia> std(MyVarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
3×1 Matrix{Float64}:
 2.23606797749979
 0.7615773105863908
 1.2083045973594573


[`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
[`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
```

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
