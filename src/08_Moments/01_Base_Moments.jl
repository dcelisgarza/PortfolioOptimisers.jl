"""
$(DocStringExtensions.TYPEDSIGNATURES)

No-op fallback for getting the view of a covariance estimator.

# Arguments

  - $(arg_dict[:ce])
  - `args...`: Optional arguments (ignored).

# Returns

  - `ce::StatsBase.CovarianceEstimator`: The original covariance estimator.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function port_opt_view(ce::StatsBase.CovarianceEstimator, ::Any,
                       args...)::StatsBase.CovarianceEstimator
    return ce
end
"""
    factory(
        ce::StatsBase.CovarianceEstimator,
        args...;
        kwargs...
    ) -> StatsBase.CovarianceEstimator

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
function factory(ce::StatsBase.CovarianceEstimator, args...;
                 kwargs...)::StatsBase.CovarianceEstimator
    return ce
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all covariance estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement covariance estimation should be subtypes of `AbstractCovarianceEstimator`.

# Interfaces

In order to implement a new covariance estimator which will work seamlessly with the library, subtype `AbstractCovarianceEstimator` with all necessary parameters---including observation weights---as part of the struct, and implement the following methods:

## Covariance and correlation

  - `Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum; kwargs...) -> MatNum`: Covariance matrix estimation.
  - `Statistics.cor(ce::AbstractCovarianceEstimator, X::MatNum; kwargs...) -> MatNum`: Correlation matrix estimation.

### Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

### Returns

  - $(ret_dict[:sigrho])

## Factory

  - `PortfolioOptimisers.factory(ce::AbstractCovarianceEstimator, w::PortfolioOptimisers.ObsWeights) -> AbstractCovarianceEstimator`: Factory method for creating instances of the estimator with new observation weights.

### Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

### Returns

  - $(ret_dict[:ce])

# Examples

We can create a dummy covariance estimator as follows:

```jldoctest
julia> struct MyCovarianceEstimator{T1} <: PortfolioOptimisers.AbstractCovarianceEstimator
           w::T1
           function MyCovarianceEstimator(w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights})
               PortfolioOptimisers.assert_nonempty_nonneg_finite_val(w, :w)
               return new{typeof(w)}(w)
           end
       end

julia> function MyCovarianceEstimator(;
                                      w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights} = nothing)
           return MyCovarianceEstimator(w)
       end
MyCovarianceEstimator

julia> function PortfolioOptimisers.factory(::MyCovarianceEstimator,
                                            w::PortfolioOptimisers.ObsWeights)
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
           w = isnothing(est.w) ? StatsBase.fweights(fill(1.0, size(X, 1))) : est.w
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

julia> PortfolioOptimisers.factory(MyCovarianceEstimator(), StatsBase.Weights([1, 2, 3]))
MyCovarianceEstimator
  w ┴ StatsBase.Weights{Int64, Int64, Vector{Int64}}: [1, 2, 3]
```

# Related

  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
  - [`AbstractMomentAlgorithm`](@ref)
"""
abstract type AbstractCovarianceEstimator <: StatsBase.CovarianceEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all variance estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement variance estimation should be subtypes of `AbstractVarianceEstimator`.

# Interfaces

In order to implement a new covariance estimator which will work seamlessly with the library, subtype `AbstractVarianceEstimator` with all necessary parameters---including observation weights---as part of the struct, and implement the following methods:

## Variance and standard deviation

  - `Statistics.var(ve::AbstractVarianceEstimator, X::MatNum; kwargs...) -> ArrNum`: Variance estimation.
  - `Statistics.std(ve::AbstractVarianceEstimator, X::MatNum; kwargs...) -> ArrNum`: Standard deviation estimation.
  - `Statistics.var(ve::AbstractVarianceEstimator, X::VecNum; kwargs...) -> Num`: Variance estimation.
  - `Statistics.std(ve::AbstractVarianceEstimator, X::VecNum; kwargs...) -> Num`: Standard deviation estimation.

### Arguments

  - $(arg_dict[:ve])

  - `X`

      + $(arg_dict[:X])
      + $(arg_dict[:Xv])

  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

### Returns

  - $(arg_dict[:X])

      + $(ret_dict[:stdvar])

  - $(arg_dict[:Xv])

      + $(ret_dict[:stdvarnum])

## Factory

  - `PortfolioOptimisers.factory(ve::AbstractVarianceEstimator, w::PortfolioOptimisers.ObsWeights) -> AbstractVarianceEstimator`: Factory method for creating instances of the estimator with new observation weights.

### Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:ow])

### Returns

  - $(ret_dict[:ve])

# Examples

We can create a dummy variance estimator as follows:

```jldoctest
julia> struct MyVarianceEstimator{T1} <: PortfolioOptimisers.AbstractVarianceEstimator
           w::T1
           function MyVarianceEstimator(w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights})
               PortfolioOptimisers.assert_nonempty_nonneg_finite_val(w, :w)
               return new{typeof(w)}(w)
           end
       end

julia> function MyVarianceEstimator(;
                                    w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights} = nothing)
           return MyVarianceEstimator(w)
       end
MyVarianceEstimator

julia> function PortfolioOptimisers.factory(::MyVarianceEstimator,
                                            w::PortfolioOptimisers.ObsWeights)
           return MyVarianceEstimator(; w = w)
       end

julia> function Statistics.var(est::MyVarianceEstimator, X::PortfolioOptimisers.MatNum;
                               dims::Int = 1, kwargs...)
           if !(dims in (1, 2))
               throw(DomainError(dims, "dims must be either 1 or 2"))
           end
           if dims == 2
               X = X'
           end
           w = isnothing(est.w) ? StatsBase.fweights(fill(1.0, size(X, 1))) : est.w
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
           w = isnothing(est.w) ? StatsBase.fweights(fill(1.0, size(X, 1))) : est.w
           X = X .* w
           sigma = sqrt.(LinearAlgebra.diag(X * X'))
           return isone(dims) ? reshape(sigma, 1, :) : reshape(sigma, :, 1)
       end

julia> function Statistics.var(est::MyVarianceEstimator, X::PortfolioOptimisers.VecNum; kwargs...)
           w = isnothing(est.w) ? StatsBase.fweights(fill(1.0, size(X, 1))) : est.w
           X = X .* w
           return mean(LinearAlgebra.diag(X' * X))
       end

julia> function Statistics.std(est::MyVarianceEstimator, X::PortfolioOptimisers.VecNum; kwargs...)
           w = isnothing(est.w) ? StatsBase.fweights(fill(1.0, size(X, 1))) : est.w
           X = X .* w
           return sqrt(mean(LinearAlgebra.diag(X' * X)))
       end

julia> var(MyVarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
1×3 Matrix{Float64}:
 5.0  0.58  1.46

julia> std(MyVarianceEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1])
1×3 Matrix{Float64}:
 2.23607  0.761577  1.2083

julia> PortfolioOptimisers.factory(MyVarianceEstimator(), StatsBase.Weights([1, 2, 3]))
MyVarianceEstimator
  w ┴ StatsBase.Weights{Int64, Int64, Vector{Int64}}: [1, 2, 3]
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type AbstractVarianceEstimator <: AbstractCovarianceEstimator end
@define_pretty_show(AbstractCovarianceEstimator)
function factory_child(v::AbstractCovarianceEstimator, args...; kwargs...)
    return factory(v, args...; kwargs...)
end
function factory_child(v::AbstractArray{<:AbstractCovarianceEstimator}, args...; kwargs...)
    return [factory_child(vi, args...; kwargs...) for vi in v]
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all expected returns estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement expected returns estimation should be subtypes of `AbstractExpectedReturnsEstimator`.

# Interfaces

In order to implement a new expected returns estimator which will work seamlessly with the library, subtype `AbstractExpectedReturnsEstimator` with all necessary parameters---including observation weights---as part of the struct, and implement the following methods:

## Expected returns

  - `Statistics.mean(me::AbstractExpectedReturnsEstimator, X::MatNum; kwargs...) -> ArrNum`: Expected returns estimation.

### Arguments

  - $(arg_dict[:me])
  - $(arg_dict[:X])
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

### Returns

  - $(ret_dict[:mu])

## Factory

  - `PortfolioOptimisers.factory(me::AbstractExpectedReturnsEstimator, w::PortfolioOptimisers.ObsWeights) -> AbstractExpectedReturnsEstimator`: Factory method for creating instances of the estimator with new observation weights.

### Arguments

  - $(arg_dict[:me])
  - $(arg_dict[:ow])

### Returns

  - $(ret_dict[:me])

# Examples

```jldoctest
julia> struct MyExpectedReturnsEstimator{T1} <:
              PortfolioOptimisers.AbstractExpectedReturnsEstimator
           w::T1
           function MyExpectedReturnsEstimator(w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights})
               PortfolioOptimisers.assert_nonempty_nonneg_finite_val(w, :w)
               return new{typeof(w)}(w)
           end
       end

julia> function MyExpectedReturnsEstimator(;
                                           w::PortfolioOptimisers.Option{<:PortfolioOptimisers.ObsWeights} = nothing)
           return MyExpectedReturnsEstimator(w)
       end
MyExpectedReturnsEstimator

julia> function PortfolioOptimisers.factory(::MyExpectedReturnsEstimator,
                                            w::PortfolioOptimisers.ObsWeights)
           return MyExpectedReturnsEstimator(; w = w)
       end

julia> function Statistics.mean(est::MyExpectedReturnsEstimator, X::PortfolioOptimisers.MatNum;
                                dims::Int = 1, kwargs...)
           if !(dims in (1, 2))
               throw(DomainError(dims, "dims must be either 1 or 2"))
           end
           if dims == 2
               X = X'
           end
           w = isnothing(est.w) ? fill(one(eltype(X)), size(X, 1)) : est.w
           X = X .* w
           mu = sum(X; dims = 1) / sum(w)
           return isone(dims) ? reshape(mu, 1, :) : reshape(mu, :, 1)
       end

julia> mean(MyExpectedReturnsEstimator(), [1.0 2.0; 0.3 0.7; 0.5 1.1]; dims = 2)
3×1 Matrix{Float64}:
 1.5
 0.5
 0.8

julia> PortfolioOptimisers.factory(MyExpectedReturnsEstimator(), StatsBase.Weights([1, 2, 3]))
MyExpectedReturnsEstimator
  w ┴ StatsBase.Weights{Int64, Int64, Vector{Int64}}: [1, 2, 3]
```

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractExpectedReturnsAlgorithm`](@ref)
"""
abstract type AbstractExpectedReturnsEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

No-op fallback for getting the view of an expected returns estimator.

# Arguments

  - $(arg_dict[:me])
  - `args...`: Optional arguments (ignored).

# Returns

  - `me::AbstractExpectedReturnsEstimator`: The original expected returns estimator.

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
function port_opt_view(me::AbstractExpectedReturnsEstimator, ::Any,
                       args...)::AbstractExpectedReturnsEstimator
    return me
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all expected returns algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a specific algorithm used by an expected returns estimator should be subtypes of `AbstractExpectedReturnsAlgorithm`.

# Interfaces

Given that these are meant to be used by expected returns estimators, there are no specific methods that need to be implemented for this abstract type. However, it serves as a marker for dispatching and organising different expected returns algorithms within the library. The interfaces should be defined at the level of the expected returns estimator that utilises these algorithms.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
abstract type AbstractExpectedReturnsAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

No-op fallback for getting the view of an expected returns algorithm.

# Arguments

  - `alg`: The expected returns algorithm.
  - `args...`: Optional arguments (ignored).

# Returns

  - `alg::AbstractExpectedReturnsAlgorithm`: The original expected returns algorithm.

# Related

  - [`AbstractExpectedReturnsAlgorithm`](@ref)
"""
function port_opt_view(alg::AbstractExpectedReturnsAlgorithm, ::Any,
                       args...)::AbstractExpectedReturnsAlgorithm
    return alg
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all moment algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types that implement a specific algorithm for moment estimation should be subtypes of `AbstractMomentAlgorithm`.

# Interfaces

Given that these are meant to be used by covariance estimators, there are no specific methods that need to be implemented for this abstract type. However, it serves as a marker for dispatching and organising different moment algorithms within the library. The interfaces should be defined at the level of the covariance estimator that utilises these algorithms.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
"""
abstract type AbstractMomentAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

`FullMoment` is used to indicate that all deviations are included in the moment estimation process.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{D} &= \\boldsymbol{X} - t
\\end{align}
```

Where:

  - $(math_dict[:Xv])
  - $(math_dict[:tgt])

# Constructors

    FullMoment() -> FullMoment

# Examples

```jldoctest
julia> FullMoment()
FullMoment()
```

# Related

  - [`AbstractMomentAlgorithm`](@ref)
  - [`SemiMoment`](@ref)
"""
struct FullMoment <: AbstractMomentAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

`SemiMoment` is used for semi-moment estimators, where only observations below a target are considered.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{D} &= \\min\\left(\\boldsymbol{X} - t,\\, 0\\right)
\\end{align}
```

Where:

  - $(math_dict[:Xv])
  - $(math_dict[:tgt])

# Constructors

    SemiMoment() -> SemiMoment

# Examples

```jldoctest
julia> SemiMoment()
SemiMoment()
```

# Related

  - [`AbstractMomentAlgorithm`](@ref)
  - [`FullMoment`](@ref)
"""
struct SemiMoment <: AbstractMomentAlgorithm end
"""
    compat_cov(
        ce::StatsBase.CovarianceEstimator,
        X::MatNum,
        [w::StatsBase.AbstractWeights];
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the covariance matrix robustly using the specified covariance estimator `ce`, data matrix `X`, and optional weights vector `w`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to `cov`.

# Returns

  - $(ret_dict[:sigma])

# Details

  - This function computes the optionally weighted covariance matrix using the provided estimator and keyword arguments.
  - Keyword arguments are only forwarded if the estimator's `cov` method accepts them (checked via `hasmethod`); otherwise the call is made with `dims` and `mean` alone. If the forwarded call throws a `MethodError` (e.g. a `kwargs...` slurp that rejects them further down its call chain), it is retried without them. Genuine errors thrown by the estimator propagate to the caller.
  - If the call throws a `MethodError`, it is retried once with a densified `Matrix(X)`.

# Related

  - [`MatNum`](@ref)
  - [`robust_cor`](@ref)
  - [`Statistics.cov`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function compat_cov(ce::StatsBase.CovarianceEstimator, X::AbstractMatrix, args...;
                    dims::Int = 1, mean = nothing, kwargs...)
    if !isempty(kwargs) &&
       hasmethod(Statistics.cov, Tuple{typeof(ce), typeof(X), typeof.(args)...},
                 (:dims, :mean, keys(kwargs)...))
        try
            return Statistics.cov(ce, X, args...; dims = dims, mean = mean, kwargs...)
        catch err
            # A method with a `kwargs...` slurp satisfies `hasmethod` but can still
            # reject the keyword arguments further down its call chain.
            if !(err isa MethodError)
                rethrow()
            end
        end
    end
    return Statistics.cov(ce, X, args...; dims = dims, mean = mean)
end
"""
    robust_cov(
        ce::StatsBase.CovarianceEstimator,
        X::MatNum,
        [w::StatsBase.AbstractWeights];
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Tries calling [`compat_cov`](@ref) and falls back to a densified `Matrix` if a `MethodError` is thrown.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to `compat_cov`.

# Returns

  - $(ret_dict[:sigma])

# Details

  - This function computes the optionally weighted covariance matrix using the provided estimator and keyword arguments.
  - If the call throws a `MethodError`, it is retried once with a densified `Matrix(X)`.

# Related

  - [`MatNum`](@ref)
  - [`compat_cov`](@ref)
  - [`Statistics.cov`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function robust_cov(ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    return try
        compat_cov(ce, X; dims = dims, mean = mean, kwargs...)
    catch err
        if !(err isa MethodError)
            rethrow()
        end
        compat_cov(ce, Matrix(X); dims = dims, mean = mean, kwargs...)
    end
end
function robust_cov(ce::StatsBase.CovarianceEstimator, X::MatNum,
                    w::StatsBase.AbstractWeights; dims::Int = 1, mean = nothing, kwargs...)
    return try
        compat_cov(ce, X, w; dims = dims, mean = mean, kwargs...)
    catch err
        if !(err isa MethodError)
            rethrow()
        end
        compat_cov(ce, Matrix(X), w; dims = dims, mean = mean, kwargs...)
    end
end
"""
    compat_cor(
        ce::StatsBase.CovarianceEstimator,
        X::MatNum,
        [w::StatsBase.AbstractWeights];
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Compute the correlation matrix robustly using the specified covariance estimator `ce`, data matrix `X`, and optional weights vector `w`.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to `cor`.

# Returns

  - $(ret_dict[:rho])

# Details

  - This function computes the optionally weighted correlation matrix using the provided estimator and keyword arguments.
  - Keyword arguments are only forwarded if the estimator's `cor` method accepts them (checked via `hasmethod`); otherwise the call is made with `dims` and `mean` alone. If the forwarded call throws a `MethodError` (e.g. a `kwargs...` slurp that rejects them further down its call chain), it is retried without them. Genuine errors thrown by the estimator propagate to the caller.
  - If the estimator defines no suitable `cor` method, the result is computed with [`robust_cov`](@ref) and converted to a correlation matrix.
  - If the call throws a `MethodError`, it is retried once with a densified `Matrix(X)`.

# Related

  - [`MatNum`](@ref)
  - [`robust_cov`](@ref)
  - [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cov/)
"""
function compat_cor(ce::StatsBase.CovarianceEstimator, X::AbstractMatrix, args...;
                    dims::Int = 1, mean = nothing, kwargs...)
    if hasmethod(Statistics.cor, Tuple{typeof(ce), typeof(X), typeof.(args)...},
                 (:dims, :mean))
        if !isempty(kwargs) &&
           hasmethod(Statistics.cor, Tuple{typeof(ce), typeof(X), typeof.(args)...},
                     (:dims, :mean, keys(kwargs)...))
            try
                return Statistics.cor(ce, X, args...; dims = dims, mean = mean, kwargs...)
            catch err
                # A method with a `kwargs...` slurp satisfies `hasmethod` but can still
                # reject the keyword arguments further down its call chain.
                if !(err isa MethodError)
                    rethrow()
                end
            end
        end
        try
            return Statistics.cor(ce, X, args...; dims = dims, mean = mean)
        catch err
            if !(err isa MethodError)
                rethrow()
            end
        end
    end
    sigma = robust_cov(ce, X, args...; dims = dims, mean = mean, kwargs...)
    if ismutable(sigma)
        StatsBase.cov2cor!(sigma, sqrt.(LinearAlgebra.diag(sigma)))
    else
        sigma = StatsBase.cov2cor(Matrix(sigma))
    end
    return sigma
end
"""
    robust_cor(
        ce::StatsBase.CovarianceEstimator,
        X::MatNum,
        [w::StatsBase.AbstractWeights];
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> MatNum

Tries calling [`compat_cor`](@ref) and falls back to a densified `Matrix` if a `MethodError` is thrown.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:X])
  - $(arg_dict[:oow])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to `compat_cor`.

# Returns

  - $(ret_dict[:rho])

# Details

  - This function computes the optionally weighted correlation matrix using the provided estimator and keyword arguments.
  - If the call throws a `MethodError`, it is retried once with a densified `Matrix(X)`.

# Related

  - [`MatNum`](@ref)
  - [`compat_cor`](@ref)
  - [`Statistics.cor`](https://juliastats.org/StatsBase.jl/stable/cor/)
"""
function robust_cor(ce::StatsBase.CovarianceEstimator, X::MatNum; dims::Int = 1,
                    mean = nothing, kwargs...)
    return try
        compat_cor(ce, X; dims = dims, mean = mean, kwargs...)
    catch err
        if !(err isa MethodError)
            rethrow()
        end
        compat_cor(ce, Matrix(X); dims = dims, mean = mean, kwargs...)
    end
end
function robust_cor(ce::StatsBase.CovarianceEstimator, X::MatNum,
                    w::StatsBase.AbstractWeights; dims::Int = 1, mean = nothing, kwargs...)
    return try
        compat_cor(ce, X, w; dims = dims, mean = mean, kwargs...)
    catch err
        if !(err isa MethodError)
            rethrow()
        end
        compat_cor(ce, Matrix(X), w; dims = dims, mean = mean, kwargs...)
    end
end
"""
    moment_window_and_weights(
        X::VecNum_MatNum,
        w::Option{<:ObsWeights},
        args...;
        dims = dims,
        kwargs...
    ) -> (VecNum_MatNum, Option{<:StatsBase.AbstractWeights})
    moment_window_and_weights(
        X::VecNum_MatNum,
        w::Option{<:ObsWeights},
        window::VecInt;
        dims = dims,
        kwargs...
    ) -> (VecNum_MatNum, Option{<:StatsBase.AbstractWeights})

Apply the observation window and resolve weights for moment estimation.

Slices `X` to the last `window` observations (if provided) and resolves the observation weights, returning the windowed data and finalised weights.

# Arguments

  - $(arg_dict[:X_Xv])
  - $(arg_dict[:oow])
  - Either:
      + $(arg_dict[:ignargs])
      + $(arg_dict[:window])
  - $(arg_dict[:dims]) Ignored if `X` is a vector.
  - $(arg_dict[:ignkwargs])

# Returns

  - `X::VecNum_MatNum`: Appropriately windowed data matrix.
  - `w::Option{<:StatsBase.AbstractWeights}`: Resolved and appropriately windowed weights.

# Details

  - If `window` is provided:
      + Gets the appropriate view of `X` given its type and the value of `dims`.
      + Calls [`nothing_scalar_array_getindex`](@ref) on `w` to resolve the windowed weights.
  - If no `window` is provided:
      + Calls [`get_observation_weights`](@ref) on `w` to resolve the weights.
  - Returns the appropriate `X` and `w`.

# Related

  - [`get_window`](@ref)
  - [`get_observation_weights`](@ref)
"""
function moment_window_and_weights(X::MatNum, w::Option{<:ObsWeights}, args...; dims = dims,
                                   kwargs...)
    w = get_observation_weights(w, X; dims = dims, kwargs...)
    return X, w
end
function moment_window_and_weights(X::VecNum, w::Option{<:ObsWeights}, args...; kwargs...)
    w = get_observation_weights(w, X; kwargs...)
    return X, w
end
function moment_window_and_weights(X::MatNum, w::Option{<:ObsWeights}, window::VecInt;
                                   dims::Int = 1, kwargs...)
    X = isone(dims) ? view(X, window, :) : view(X, :, window)
    w = nothing_scalar_array_getindex(w, window)
    w = get_observation_weights(w, X; dims = dims, kwargs...)
    return X, w
end
function moment_window_and_weights(X::VecNum, w::Option{<:ObsWeights}, window::VecInt;
                                   kwargs...)
    X = view(X, window)
    w = nothing_scalar_array_getindex(w, window)
    w = get_observation_weights(w, X; kwargs...)
    return X, w
end
"""
    demean_returns(X::MatNum, me::AbstractExpectedReturnsEstimator; dims::Int = 1, mean = nothing,
                   kwargs...) -> MatNum

Demeans the returns in `X` using the expected returns estimator `me` or if provided, a `mean` array.

# Arguments

  - $(arg_dict[:X])
  - $(arg_dict[:me])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments for the expected returns estimator.

# Returns

  - `MatNum`: The demeaned returns matrix.

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
"""
function demean_returns(X::MatNum, me::AbstractExpectedReturnsEstimator; dims::Int = 1,
                        mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(me, X; dims = dims, kwargs...) : mean
    return X .- mu
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Shared preamble for windowed moment estimators (matrix input).

Resolves the window specification, subsets `X` (and `iv`) to the selected observations,
rebinds observation weights to the window, and builds a weight-updated copy of `est` via
[`factory`](@ref). Whenever a window is given — an `Int` (which resolves to a range) or an
explicit index vector — `iv` is subset to the same rows, or columns when `dims = 2`, so it
stays aligned with the windowed returns. Only `window = nothing`, which resolves to a
`Colon`, leaves `iv` unchanged.

# Arguments

  - `est`: Wrapped moment estimator to be cloned with updated weights.
  - `w`: Optional observation weights applied after windowing.
  - `window`: Window specification — `nothing` (full data), an `Int` (last `window`
    observations), or a `VecInt` of explicit row/column indices.
  - `X`: Data matrix of asset returns.
  - `iv`: Optional instrument variable matrix; subsetted to the window when `window` is a
    `VecInt`.
  - `dims`: Observation dimension — 1 for rows (default), 2 for columns.
  - `kwargs...`: Passed through to [`moment_window_and_weights`](@ref).

# Returns

  - `(inner, X, iv)`: Weight-updated estimator, windowed returns matrix, and (possibly
    subsetted) instrument variable matrix.

# Related

  - [`get_window`](@ref)
  - [`moment_window_and_weights`](@ref)
  - [`factory`](@ref)
  - [`@windowed_estimator`](@ref)
  - [`WindowedExpectedReturns`](@ref)
  - [`WindowedCovariance`](@ref)
  - [`WindowedVariance`](@ref)
  - [`WindowedCoskewness`](@ref)
  - [`WindowedCokurtosis`](@ref)
"""
function windowed_preamble(est, w::Option{<:ObsWeights}, window::Option{<:Int_VecInt},
                           X::MatNum; iv::Option{<:MatNum} = nothing, dims::Int = 1,
                           kwargs...)
    win = get_window(window, X, dims)
    X, w_new = moment_window_and_weights(X, w, win; dims = dims, kwargs...)
    inner = factory(est, w_new)
    if !isnothing(iv) && isa(win, VecInt)
        iv = isone(dims) ? view(iv, win, :) : view(iv, :, win)
    end
    return inner, X, iv
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Shared preamble for windowed moment estimators (vector input).

Resolves the window specification, subsets `X` to the selected observations, rebinds
observation weights to the window, and builds a weight-updated copy of `est` via
[`factory`](@ref).

# Arguments

  - `est`: Wrapped moment estimator to be cloned with updated weights.
  - `w`: Optional observation weights applied after windowing.
  - `window`: Window specification — `nothing` (full data), an `Int` (last `window`
    observations), or a `VecInt` of explicit indices.
  - `X`: Data vector of returns.

# Returns

  - `(inner, X)`: Weight-updated estimator and windowed returns vector.

# Related

  - [`get_window`](@ref)
  - [`moment_window_and_weights`](@ref)
  - [`factory`](@ref)
  - [`@windowed_estimator`](@ref)
  - [`WindowedVariance`](@ref)
"""
function windowed_preamble(est, w::Option{<:ObsWeights}, window::Option{<:Int_VecInt},
                           X::VecNum)
    win = get_window(window, X)
    X, w_new = moment_window_and_weights(X, w, win)
    inner = factory(est, w_new)
    return inner, X
end

# ---------------------------------------------------------------------------
# @windowed_estimator — one declaration per windowed moment estimator (ADR 0039)
# ---------------------------------------------------------------------------
"""
    WINDOWED_ESTIMATOR_KEYS

Assignment keys recognised in a [`@windowed_estimator`](@ref) body, besides the single
`field::Type = default` declaration. Anything else is rejected at macro-expansion time with
a [`did_you_mean`](@ref) suggestion, so a mistyped key cannot silently produce a malformed
docstring or a missing forwarding method.

# Related

  - [`@windowed_estimator`](@ref)
"""
const WINDOWED_ESTIMATOR_KEYS = (:noun, :forward, :doctest)
"""
    WINDOWED_ESTIMATOR_INPUTS

Input types a [`@windowed_estimator`](@ref) `forward` entry may declare. `MatNum` generates
the matrix forwarder (threading `dims` and `iv` through [`windowed_preamble`](@ref)),
`VecNum` the vector forwarder.

# Related

  - [`@windowed_estimator`](@ref)
"""
const WINDOWED_ESTIMATOR_INPUTS = (:MatNum, :VecNum)
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Throw a uniform, expansion-time [`ArgumentError`](https://docs.julialang.org/en/v1/base/base/#Core.ArgumentError) for a malformed [`@windowed_estimator`](@ref) declaration.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_estimator_error(msg::AbstractString)
    return throw(ArgumentError("@windowed_estimator: " * msg))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Validate that `key` names an entry of `dict`, appending a [`did_you_mean`](@ref) suggestion
to the error when it does not. `what` names the table in the message.

# Related

  - [`@windowed_estimator`](@ref)
  - [`did_you_mean`](@ref)
"""
function windowed_estimator_check_key(key::Symbol, dict::AbstractDict, what::AbstractString)
    if !haskey(dict, key)
        windowed_estimator_error("`$(key)` is not a `$(what)` key" *
                                 windowed_estimator_suggest(key, keys(dict)) *
                                 ".")
    end
    return key
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Suggest the nearest `candidates` entry to a mistyped [`@windowed_estimator`](@ref) key.

Wraps [`did_you_mean`](@ref) in a looser scoped configuration than the global default:
Damerau-Levenshtein (so a transposed pair costs one edit, not two) at `min_score = 0.5`. The
strict global default exists to keep near-miss probes from echoing real *asset names* back to
the caller (ADR 0026); that boundary does not apply here, because the candidates are
compile-time constants — block keys and `field_dict`/`ret_dict` names — with nothing to leak.
At the default `0.7` under plain Levenshtein, short keys never match: `nuon` scores 0.5
against `noun`, so the suggestion would be dead code.

# Related

  - [`@windowed_estimator`](@ref)
  - [`did_you_mean`](@ref)
  - [`with_string_distance`](@ref)
"""
function windowed_estimator_suggest(key, candidates)
    return with_string_distance(; dist = StringDistances.DamerauLevenshtein(),
                                min_score = 0.5) do
        return did_you_mean(string(key), string.(collect(candidates)))
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Parse the `field::Type = default` line of a [`@windowed_estimator`](@ref) body into the
inner estimator's field name, its declared type, and its keyword-constructor default.

The field name doubles as the [`field_dict`](@ref) key for the generated field docstring and
as the argument name of every generated forwarding method, so it must follow the library's
naming convention (`me`, `ce`, `ve`, `ske`, `kte`).

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_parse_field(ex)
    if !(Meta.isexpr(ex, :(=)) &&
         Meta.isexpr(ex.args[1], :(::)) &&
         length(ex.args[1].args) == 2)
        windowed_estimator_error("the inner estimator must be declared as `field::Type = default`; got `$(ex)`.")
    end
    name, type = ex.args[1].args
    if !isa(name, Symbol)
        windowed_estimator_error("the inner estimator's field name must be a symbol; got `$(name)`.")
    end
    windowed_estimator_check_key(name, field_dict, "field_dict")
    return name, type, ex.args[2]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Parse one `forward` entry of a [`@windowed_estimator`](@ref) body — `generic(::MatNum; mean) => :ret_key` — into the generic being forwarded, its input type, whether it names a
`mean` keyword, and the [`ret_dict`](@ref) keys documenting its return values.

Naming `mean` in the mini-signature is what keeps it out of the forwarded `kwargs...`, where
it would otherwise leak into [`windowed_preamble`](@ref).

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_parse_forward(ex)
    if !(Meta.isexpr(ex, :call) && length(ex.args) == 3 && ex.args[1] === :(=>))
        windowed_estimator_error("each `forward` entry must read `generic(::MatNum[; mean]) => :ret_key` (or `=> (:k1, :k2)` for a tuple return); got `$(ex)`.")
    end
    sig, rets = ex.args[2], ex.args[3]
    if !Meta.isexpr(sig, :call)
        windowed_estimator_error("the left of a `forward` entry must be a call; got `$(sig)`.")
    end
    gen = sig.args[1]
    input, has_mean = nothing, false
    for arg in sig.args[2:end]
        if Meta.isexpr(arg, :parameters)
            for kw in arg.args
                if kw !== :mean
                    windowed_estimator_error("`mean` is the only keyword a `forward` entry may name; got `$(kw)`.")
                end
                has_mean = true
            end
        elseif Meta.isexpr(arg, :(::)) && length(arg.args) == 1
            if !isnothing(input)
                windowed_estimator_error("a `forward` entry takes exactly one positional argument type; got `$(sig)`.")
            end
            input = arg.args[1]
        else
            windowed_estimator_error("unexpected argument `$(arg)` in `forward` entry `$(sig)`.")
        end
    end
    if !(input in WINDOWED_ESTIMATOR_INPUTS)
        windowed_estimator_error("a `forward` entry's input type must be one of $(WINDOWED_ESTIMATOR_INPUTS); got `$(input)`.")
    end
    raw = Meta.isexpr(rets, :tuple) ? rets.args : Any[rets]
    keys_ = Symbol[]
    for r in raw
        if !(isa(r, QuoteNode) && isa(r.value, Symbol))
            windowed_estimator_error("`ret_dict` keys must be quoted symbols; got `$(r)`.")
        end
        push!(keys_, windowed_estimator_check_key(r.value, ret_dict, "ret_dict"))
    end
    return gen, input, has_mean, keys_
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Render the `generic(field::Name, X::Input)` reference used to cross-link one generated
forwarding method from the type's and its siblings' `# Related` sections.

Keyword arguments are deliberately omitted: Documenter resolves an `@ref` by positional
method signature, and the two positional types already identify the method uniquely.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_method_ref(gen, field::Symbol, name::Symbol, input::Symbol)
    return "[`$(gen)($(field)::$(name), X::$(input))`](@ref)"
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the docstring for one generated forwarding method as an interpolation AST.

Returning `Expr(:string, ...)` rather than a `String` is load-bearing: it keeps
[`arg_dict`](@ref) and [`ret_dict`](@ref) lookups as live parts of the `DocStr`, exactly as
a hand-written `\$(arg_dict[:dims])` would be.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_method_doc(gen, field::Symbol, name::Symbol, input::Symbol,
                             has_mean::Bool, ret_keys::Vector{Symbol}, noun::AbstractString,
                             siblings::Vector{String})
    lc = lowercasefirst(noun)
    is_mat = input === :MatNum
    sig = if is_mat
        "    $(gen)($(field)::$(name), X::MatNum; dims::Int = 1, " *
        (has_mean ? "mean = nothing, " : "") *
        "iv::Option{<:MatNum} = nothing, kwargs...)"
    else
        "    $(gen)($(field)::$(name), X::VecNum" *
        (has_mean ? "; mean = nothing" : "") *
        ")"
    end
    # The summary names the *generic*, not the type's noun: `std` on a WindowedVariance
    # computes a standard deviation, not a variance.
    parts = Any["\n", sig,
                "\n\nCompute `$(gen)` over a rolling or indexed observation window ($(is_mat ? "matrix" : "vector") input).\n\nThis method selects a window of observations from `X` (and applies observation weights if specified), then delegates to the underlying $(lc) estimator.\n\n# Arguments\n\n  - `$(field)`: Windowed $(lc) estimator.\n  - `X`: Data $(is_mat ? "matrix of asset returns (observations × assets)" : "vector of returns").\n"]
    if is_mat
        push!(parts, "  - ", :(arg_dict[:dims]), "\n")
    end
    if has_mean
        push!(parts,
              "  - `mean`: Optional pre-computed mean passed to the underlying estimator.\n")
    end
    if is_mat
        push!(parts, "  - ", :(arg_dict[:oiv]),
              "\n  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.\n")
    end
    push!(parts, "\n# Returns\n")
    for k in ret_keys
        push!(parts, "\n  - ", :(ret_dict[$(QuoteNode(k))]))
    end
    push!(parts, "\n\n# Related\n\n  - [`$(name)`](@ref)\n")
    for s in siblings
        push!(parts, "  - ", s, "\n")
    end
    push!(parts, "  - [`windowed_preamble`](@ref)\n")
    return Expr(:string, parts...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build one generated forwarding method: resolve the window via [`windowed_preamble`](@ref),
then delegate to the inner estimator's own method.

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_method_def(gen, field::Symbol, name::Symbol, input::Symbol,
                             has_mean::Bool)
    self = Expr(:(::), field, name)
    inner = Expr(:., field, QuoteNode(field))
    w = Expr(:., field, QuoteNode(:w))
    window = Expr(:., field, QuoteNode(:window))
    return if input === :MatNum
        kws = Any[Expr(:kw, :(dims::Int), 1)]
        if has_mean
            push!(kws, Expr(:kw, :mean, nothing))
        end
        push!(kws, Expr(:kw, :(iv::Option{<:MatNum}), nothing), :(kwargs...))
        call = Any[:(dims = dims)]
        if has_mean
            push!(call, :(mean = mean))
        end
        push!(call, :(iv = iv), :(kwargs...))
        Expr(:function, Expr(:call, gen, Expr(:parameters, kws...), self, :(X::MatNum)),
             quote
                 inner, X, iv = windowed_preamble($inner, $w, $window, X; iv = iv,
                                                  dims = dims, kwargs...)
                 return $(Expr(:call, gen, Expr(:parameters, call...), :inner, :X))
             end)
    else
        kws = has_mean ? Any[Expr(:kw, :mean, nothing)] : Any[]
        call = has_mean ? Any[:(mean = mean)] : Any[]
        Expr(:function, Expr(:call, gen, Expr(:parameters, kws...), self, :(X::VecNum)),
             quote
                 inner, X = windowed_preamble($inner, $w, $window, X)
                 return $(Expr(:call, gen, Expr(:parameters, call...), :inner, :X))
             end)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the type docstring of a generated windowed estimator as an interpolation AST, keeping
`DocStringExtensions` abbreviations and dictionary lookups live (see
[`windowed_method_doc`](@ref)).

# Related

  - [`@windowed_estimator`](@ref)
"""
function windowed_type_doc(name::Symbol, super, field::Symbol, ftype, default,
                           noun::AbstractString, doctest::AbstractString,
                           methods::Vector{String})
    lc = lowercasefirst(noun)
    inner_ref = Meta.isexpr(default, :call) ? default.args[1] : default
    parts = Any["\n", :(DocStringExtensions.TYPEDEF),
                "\n\n$(noun) estimator that restricts computation to a rolling or indexed observation window.\n\n`$(name)` wraps another $(lc) estimator and applies it to a subset of observations defined by a window and/or custom observation weights. This enables time-varying or recency-weighted $(lc) estimation.\n\n# Fields\n\n",
                :(DocStringExtensions.FIELDS),
                "\n\n# Constructors\n\n    $(name)(;\n        $(field)::$(ftype) = $(default),\n        w::Option{<:ObsWeights} = nothing,\n        window::Option{<:Int_VecInt} = nothing\n    ) -> $(name)\n\nKeywords correspond to the struct's fields.\n\n## Validation\n\n  - ",
                :(val_dict[:oow]),
                "\n  - If `window` is provided, it must be nonempty, nonnegative, and finite.\n\n## Propagated parameters\n\nWhen [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:\n\n  - `$(field)`: Recursively updated via [`factory`](@ref).\n  - `w`: Replaced with the incoming [`ObsWeights`](@ref).\n\n## View parameters\n\nWhen [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:\n\n  - `$(field)`: Recursively viewed via [`port_opt_view`](@ref).\n\n# Examples\n\n```jldoctest\n$(strip(doctest))\n```\n\n# Related\n\n  - [`$(super)`](@ref)\n  - [`$(inner_ref)`](@ref)\n"]
    for m in methods
        push!(parts, "  - ", m, "\n")
    end
    push!(parts,
          "  - [`factory`](@ref)\n  - [`port_opt_view`](@ref)\n  - [`windowed_preamble`](@ref)\n")
    return Expr(:string, parts...)
end
"""
    @windowed_estimator Name <: Super begin
        field::FieldType = Default()
        noun    = "Noun"
        forward = [generic(::MatNum; mean) => :ret_key, ...]
        doctest = \"\"\"...\"\"\"
    end

Declare a windowed moment estimator: a wrapper that restricts an inner moment estimator to a
sub-window of observations and rebinds observation weights to that window, leaving the inner
estimator's semantics untouched.

One invocation emits the whole family member — the [`@propagatable`](@ref) `@concrete`
struct (inner estimator tagged `@fprop @vprop`, `w` tagged `@wprop`, plus `window`), both
constructors with their validation, one forwarding method per `forward` entry, the `export`,
and every docstring.

Five nominal types exist rather than one parametric `Windowed{E}` because each answers a
different generic and must subtype a different abstract estimator — `AbstractCovarianceEstimator`,
`CoskewnessEstimator`, and the rest are load-bearing for dispatch across the library, and a
Julia struct's supertype cannot depend on a type parameter. This macro is what keeps the five
in sync; see ADR 0039.

# Body

  - `field::FieldType = Default()`: the inner estimator. The field name is also its
    [`field_dict`](@ref) key and the argument name of every generated method, so it must
    follow the library convention (`me`, `ce`, `ve`, `ske`, `kte`).
  - `noun`: capitalised noun phrase naming the moment, e.g. `"Expected returns"`. Drives all
    generated prose.
  - `forward`: one mini-signature per generic to forward, paired with the
    [`ret_dict`](@ref) key(s) documenting its return values. Naming `mean` in the
    mini-signature emits it as a named keyword instead of letting it ride in `kwargs...`,
    where it would leak into [`windowed_preamble`](@ref).
  - `doctest`: the body of the `jldoctest` block for the `# Examples` section, without its
    fences.

Unknown keys, malformed `forward` entries, and unknown `field_dict`/`ret_dict` keys are
rejected at macro-expansion time with a [`did_you_mean`](@ref) suggestion.

# Examples

```julia
@windowed_estimator WindowedVariance <: AbstractVarianceEstimator begin
    ve::AbstractVarianceEstimator = SimpleVariance()
    noun    = "Variance"
    forward = [Statistics.var(::MatNum; mean) => :vararr,
               Statistics.var(::VecNum; mean) => :varnum]
    doctest = \"\"\"
    julia> WindowedVariance()
    ...
    \"\"\"
end
```

# Related

  - [`windowed_preamble`](@ref)
  - [`@propagatable`](@ref)
  - [`WindowedExpectedReturns`](@ref)
  - [`WindowedCovariance`](@ref)
  - [`WindowedVariance`](@ref)
  - [`WindowedCoskewness`](@ref)
  - [`WindowedCokurtosis`](@ref)
"""
macro windowed_estimator(head, body)
    if !(Meta.isexpr(head, :<:) && isa(head.args[1], Symbol))
        windowed_estimator_error("the header must read `Name <: Super`; got `$(head)`.")
    end
    name, super = head.args
    if !Meta.isexpr(body, :block)
        windowed_estimator_error("the declaration body must be a `begin ... end` block.")
    end
    field, ftype, default = nothing, nothing, nothing
    noun, forward, doctest = nothing, nothing, nothing
    for stmt in body.args
        if isa(stmt, LineNumberNode)
            continue
        end
        if !Meta.isexpr(stmt, :(=))
            windowed_estimator_error("every line of the body must be an assignment; got `$(stmt)`.")
        end
        lhs = stmt.args[1]
        if Meta.isexpr(lhs, :(::))
            if !isnothing(field)
                windowed_estimator_error("exactly one `field::Type = default` line is allowed.")
            end
            field, ftype, default = windowed_parse_field(stmt)
        elseif lhs === :noun
            noun = stmt.args[2]
        elseif lhs === :forward
            if !Meta.isexpr(stmt.args[2], :vect)
                windowed_estimator_error("`forward` must be a vector of `generic(::Input) => :ret_key` entries.")
            end
            forward = stmt.args[2].args
        elseif lhs === :doctest
            doctest = stmt.args[2]
        else
            windowed_estimator_error("`$(lhs)` is not a recognised key" *
                                     windowed_estimator_suggest(lhs,
                                                                WINDOWED_ESTIMATOR_KEYS) *
                                     ".")
        end
    end
    for (val, key) in
        ((field, "field::Type = default"), (noun, "noun"), (forward, "forward"),
         (doctest, "doctest"))
        isnothing(val) && windowed_estimator_error("missing required `$(key)` declaration.")
    end
    if !isa(noun, AbstractString) || !isa(doctest, AbstractString)
        windowed_estimator_error("`noun` and `doctest` must be string literals.")
    end
    if isempty(forward)
        windowed_estimator_error("`forward` must declare at least one generic.")
    end
    specs = [windowed_parse_forward(f) for f in forward]
    refs = [windowed_method_ref(gen, field, name, input) for (gen, input, _, _) in specs]
    defs = Any[]
    for (i, (gen, input, has_mean, ret_keys)) in pairs(specs)
        siblings = [r for (j, r) in pairs(refs) if j != i]
        push!(defs,
              Expr(:macrocall, GlobalRef(Core, Symbol("@doc")), LineNumberNode(@__LINE__),
                   windowed_method_doc(gen, field, name, input, has_mean, ret_keys, noun,
                                       siblings),
                   windowed_method_def(gen, field, name, input, has_mean)))
    end
    structexpr = Expr(:struct, false, Expr(:<:, name, super),
                      Expr(:block, Expr(:string, :(field_dict[$(QuoteNode(field))])),
                           Expr(:macrocall, Symbol("@fprop"), LineNumberNode(@__LINE__),
                                Expr(:macrocall, Symbol("@vprop"),
                                     LineNumberNode(@__LINE__), field)),
                           Expr(:string, :(field_dict[:oow])),
                           Expr(:macrocall, Symbol("@wprop"), LineNumberNode(@__LINE__),
                                :w),
                           "Window specification: an integer (last `window` observations) or a vector of indices.",
                           :window,
                           Expr(:function,
                                Expr(:call, name, Expr(:(::), field, ftype),
                                     :(w::Option{<:ObsWeights}),
                                     :(window::Option{<:Int_VecInt})),
                                quote
                                    assert_nonempty_nonneg_finite_val(w, :w)
                                    assert_nonempty_nonneg_finite_val(window, :window)
                                    return $(Expr(:call,
                                                  Expr(:curly, :new,
                                                       Expr(:call, :typeof, field),
                                                       :(typeof(w)), :(typeof(window))),
                                                  field, :w, :window))
                                end)))
    kwctor = Expr(:function,
                  Expr(:(::),
                       Expr(:call, name,
                            Expr(:parameters, Expr(:kw, Expr(:(::), field, ftype), default),
                                 Expr(:kw, :(w::Option{<:ObsWeights}), nothing),
                                 Expr(:kw, :(window::Option{<:Int_VecInt}), nothing))),
                       name),
                  Expr(:block, Expr(:return, Expr(:call, name, field, :w, :window))))
    return esc(Expr(:block,
                    Expr(:macrocall, GlobalRef(Core, Symbol("@doc")),
                         LineNumberNode(@__LINE__),
                         windowed_type_doc(name, super, field, ftype, default, noun,
                                           doctest, refs),
                         Expr(:macrocall, Symbol("@propagatable"),
                              LineNumberNode(@__LINE__),
                              Expr(:macrocall, Symbol("@concrete"),
                                   LineNumberNode(@__LINE__), structexpr))), kwctor,
                    defs..., Expr(:export, name)))
end

export FullMoment, SemiMoment
