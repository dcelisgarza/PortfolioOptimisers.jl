"""
    factory(ce::StatsBase.CovarianceEstimator, args...; kwargs...)

Fallback for covariance estimator factory methods.

# Arguments

  - $(arg_dict[:ce])
  - `args...`: Optional arguments (ignored).
  - `kwargs...`: Optional keyword arguments (ignored).

# Returns

  - `ce::StatsBase.CovarianceEstimator`: The original covariance estimator.

# Related

  - [`factory`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function factory(ce::StatsBase.CovarianceEstimator, args...; kwargs...)
    return ce
end
"""
    abstract type AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator end

Abstract supertype for all covariance estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement covariance estimation should be subtypes of `AbstractCovarianceEstimator`.

# Interfaces

In order to implement a new covariance estimator which will work seamlessly with the library, subtype `AbstractCovarianceEstimator` with all necessary parameters---including observation weights---as part of the struct, and implement the following methods:

## Covariance and correlation

  - `Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum; kwargs...)`: Covariance matrix estimation.
  - `Statistics.cor(ce::AbstractCovarianceEstimator, X::MatNum; kwargs...)`: Correlation matrix estimation.

### Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

### Returns

  - `sigma::MatNum`: Covariance matrix.

## Factory

  - `factory(ce::AbstractCovarianceEstimator, w::StatsBase.AbstractWeights)`: Factory method for creating instances of the estimator with new observation weights.

### Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

### Returns

  - $(arg_dict[:nce])

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
MyCovarianceEstimator

julia> function factory(::MyCovarianceEstimator, w::StatsBase.AbstractWeights)
           return MyCovarianceEstimator(; w = w)
       end
factory (generic function with 1 method)

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

Abstract supertype for all variance estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement variance estimation should be subtypes of `AbstractVarianceEstimator`.

# Interfaces

In order to implement a new covariance estimator which will work seamlessly with the library, subtype `AbstractVarianceEstimator` with all necessary parameters---including observation weights---as part of the struct, and implement the following methods:

## Variance and standard deviation

  - `Statistics.var(ve::AbstractVarianceEstimator, X::MatNum; kwargs...)`: Variance estimation.
  - `Statistics.std(ve::AbstractVarianceEstimator, X::MatNum; kwargs...)`: Standard deviation estimation.
  - `Statistics.var(ve::AbstractVarianceEstimator, X::VecNum; kwargs...)`: Variance estimation.
  - `Statistics.std(ve::AbstractVarianceEstimator, X::VecNum; kwargs...)`: Standard deviation estimation.

### Arguments

  - $(arg_dict[:ve])

  - `X`

      + $(arg_dict[:X])
      + $(arg_dict[:Xv])

  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

### Returns

  - $(arg_dict[:X])

      + `val::MatNum`: Variance or standard deviation vector of `X`, reshaped to be consistent with the dimension along which the value is computed.

  - $(arg_dict[:Xv])

      + `val::VecNum`: Variance or standard deviation of `X`.

## Factory

  - `factory(ve::AbstractVarianceEstimator, w::StatsBase.AbstractWeights)`: Factory method for creating instances of the estimator with new observation weights.

### Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:ow])

### Returns

  - $(arg_dict[:nve])

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
           sigma = LinearAlgebra.diag(X * X')
           return isone(dims) ? reshape(sigma, 1, :) : reshape(sigma, :, 2)
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
           sigma = sqrt.(LinearAlgebra.diag(X * X'))
           return isone(dims) ? reshape(sigma, 1, :) : reshape(sigma, :, 1)
       end

julia> function Statistics.var(est::MyVarianceEstimator, X::PortfolioOptimisers.VecNum; kwargs...)
           w = ifelse(isnothing(est.w), StatsBase.fweights(fill(1.0, size(X, 1))), est.w)
           X = X .* w
           return mean(LinearAlgebra.diag(X' * X))
       end

julia> function Statistics.std(est::MyVarianceEstimator, X::PortfolioOptimisers.VecNum; kwargs...)
           w = ifelse(isnothing(est.w), StatsBase.fweights(fill(1.0, size(X, 1))), est.w)
           X = X .* w
           return sqrt(mean(LinearAlgebra.diag(X' * X)))
       end

julia> var(MyVarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
1×3 Matrix{Float64}:
 5.0  0.58  1.46

julia> std(MyVarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
1×3 Matrix{Float64}:
 2.23607  0.761577  1.2083
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type AbstractVarianceEstimator <: AbstractCovarianceEstimator end
@define_pretty_show(AbstractCovarianceEstimator)
"""
    abstract type AbstractExpectedReturnsEstimator <: AbstractEstimator end

Abstract supertype for all expected returns estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement expected returns estimation should be subtypes of `AbstractExpectedReturnsEstimator`.

# Interfaces

In order to implement a new expected returns estimator which will work seamlessly with the library, subtype `AbstractExpectedReturnsEstimator` with all necessary parameters---including observation weights---as part of the struct, and implement the following methods:

## Expected returns

  - `Statistics.mean(me::AbstractExpectedReturnsEstimator, X::MatNum; kwargs...)`: Expected returns estimation.

### Arguments

    - $(arg_dict[:me])
    - $(arg_dict[:X])
    - `kwargs...`: Additional keyword arguments passed to the mean estimator.

### Returns

    - `val::VecNum`: Expected returns vector of `X`, reshaped to be consistent with the dimension along which the value is computed.

## Factory

  - `factory(me::AbstractExpectedReturnsEstimator, w::StatsBase.AbstractWeights)`: Factory method for creating instances of the estimator with new observation weights.

### Arguments

    - $(arg_dict[:me])
    - $(arg_dict[:ow])

### Returns

    - $(arg_dict[:nme])

# Examples

```jldoctest
julia> struct MyExpectedReturnsEstimator{T1} <:
              PortfolioOptimisers.AbstractExpectedReturnsEstimator
           w::T1
           function MyExpectedReturnsEstimator(w::PortfolioOptimisers.Option{<:StatsBase.AbstractWeights})
               if !isnothing(w) && isempty(w)
                   throw(IsEmptyError("`w` cannot be an empty weights object"))
               end
               return new{typeof(w)}(w)
           end
       end

julia> function MyExpectedReturnsEstimator(;
                                           w::PortfolioOptimisers.Option{<:StatsBase.AbstractWeights} = nothing)
           return MyExpectedReturnsEstimator(w)
       end
MyExpectedReturnsEstimator

julia> function factory(::MyExpectedReturnsEstimator, w::StatsBase.AbstractWeights)
           return MyExpectedReturnsEstimator(; w = w)
       end
factory (generic function with 1 method)

julia> function Statistics.mean(est::MyExpectedReturnsEstimator, X::PortfolioOptimisers.MatNum;
                                dims::Int = 1, kwargs...)
           if !(dims in (1, 2))
               throw(DomainError(dims, "dims must be either 1 or 2"))
           end
           if dims == 2
               X = X'
           end
           w = ifelse(isnothing(est.w), fill(one(eltype(X)), size(X, 1)), est.w)
           X = X .* w
           mu = sum(X; dims = 1) / sum(w)
           return isone(dims) ? reshape(mu, 1, :) : reshape(mu, :, 1)
       end

julia> mean(MyExpectedReturnsEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1]; dims = 2)
3×1 Matrix{Float64}:
 1.5
 0.5
 0.8
```

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractExpectedReturnsAlgorithm`](@ref)
"""
abstract type AbstractExpectedReturnsEstimator <: AbstractEstimator end
"""
    abstract type AbstractExpectedReturnsAlgorithm <: AbstractAlgorithm end

Abstract supertype for all expected returns algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a specific algorithm used by an expected returns estimator should be subtypes of `AbstractExpectedReturnsAlgorithm`.

# Interfaces

Given that these are meant to be used by expected returns estimators, there are no specific methods that need to be implemented for this abstract type. However, it serves as a marker for dispatching and organizing different expected returns algorithms within the library. The interfaces should be defined at the level of the expected returns estimator that utilises these algorithms.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
abstract type AbstractExpectedReturnsAlgorithm <: AbstractAlgorithm end
"""
    abstract type AbstractMomentAlgorithm <: AbstractAlgorithm end

Abstract supertype for all moment algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a specific algorithm for moment estimation should be subtypes of `AbstractMomentAlgorithm`.

# Interfaces

Given that these are meant to be used by covariance estimators, there are no specific methods that need to be implemented for this abstract type. However, it serves as a marker for dispatching and organizing different moment algorithms within the library. The interfaces should be defined at the level of the covariance estimator that utilises these algorithms.

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

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - `mean`: Optional mean array to use for centering.
  - `kwargs...`: Additional keyword arguments passed to `cov`.

# Returns

  - `sigma::MatNum`: Covariance matrix.

# Details

  - This function attempts to compute the optionally weighted covariance matrix using the provided estimator and keyword arguments.
  - If an error occurs (e.g., due to unsupported keyword arguments), it retries with a reduced set of arguments for compatibility. This ensures robust covariance estimation across different estimator types.

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

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - `mean`: Optional mean array to use for centering.
  - `kwargs...`: Additional keyword arguments passed to `cor`.

# Returns

  - `rho::MatNum`: Correlation matrix.

# Details

  - This function attempts to compute the optionally weighted correlation matrix using the provided estimator and keyword arguments.
  - If an error occurs, it falls back to computing the optionally weighted covariance matrix and then converts it to a correlation matrix.
  - If that also errors, it tries again with [`robust_cov`](@ref) and converts the result to a correlation matrix. This ensures robust correlation estimation across different estimator types.

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
            StatsBase.cov2cor!(sigma, sqrt.(LinearAlgebra.diag(sigma)))
        else
            sigma = StatsBase.cov2cor(Matrix(sigma))
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
            StatsBase.cov2cor!(sigma, sqrt.(LinearAlgebra.diag(sigma)))
        else
            sigma = StatsBase.cov2cor(Matrix(sigma))
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
            StatsBase.cov2cor!(sigma, sqrt.(LinearAlgebra.diag(sigma)))
        else
            sigma = StatsBase.cov2cor(Matrix(sigma))
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
            StatsBase.cov2cor!(sigma, sqrt.(LinearAlgebra.diag(sigma)))
        else
            sigma = StatsBase.cov2cor(Matrix(sigma))
        end
        sigma
    end
    =#
end

export Full, Semi
