"""
    GeneralWeightedCovariance{T1, T2} <: AbstractCovarianceEstimator

A flexible covariance estimator for PortfolioOptimisers.jl supporting arbitrary covariance estimators and optional observation weights.

`GeneralWeightedCovariance` allows users to specify both the covariance estimation method and optional observation weights. This enables robust and extensible covariance estimation workflows.

# Fields

  - `ce`: Covariance estimator.
  - `w`: Optional weights for each observation. If `nothing`, the estimator is unweighted.

# Constructor

    GeneralWeightedCovariance(; ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                                 corrected = true),
                                w::Union{Nothing, <:AbstractWeights} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - If `w` is provided, it must not be empty.

# Examples

```jldoctest
julia> using StatsBase

julia> gwc = GeneralWeightedCovariance()
GeneralWeightedCovariance
  ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
   w | nothing

julia> w = Weights([0.1, 0.2, 0.7]);

julia> gwc = GeneralWeightedCovariance(; w = w)
GeneralWeightedCovariance
  ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
   w | StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.1, 0.2, 0.7]
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
struct GeneralWeightedCovariance{T1, T2} <: AbstractCovarianceEstimator
    ce::T1
    w::T2
end
function GeneralWeightedCovariance(;
                                   ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                                  corrected = true),
                                   w::Union{Nothing, <:AbstractWeights} = nothing)
    if isa(w, AbstractWeights)
        @argcheck(!isempty(w))
    end
    return GeneralWeightedCovariance(ce, w)
end

"""
    cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the covariance matrix using a [`GeneralWeightedCovariance`](@ref) estimator.

This method dispatches to [`robust_cov`](@ref), using the specified covariance estimator and optional observation weights stored in `ce`. If no weights are provided, the unweighted covariance is computed; otherwise, the weighted covariance is used.

# Arguments

  - `ce`: Covariance estimator containing the method and optional weights.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the covariance.
  - `mean`: Optional mean vector to use for centering.
  - `kwargs...`: Additional keyword arguments passed to [`robust_cov`](@ref).

# Returns

  - Covariance matrix as computed by the estimator and optional weights.

# Related

  - [`robust_cov`](@ref)
  - [`cor(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
    return if isnothing(ce.w)
        robust_cov(ce.ce, X; dims = dims, mean = mean, kwargs...)
    else
        robust_cov(ce.ce, X, ce.w; dims = dims, mean = mean, kwargs...)
    end
end

"""
    cor(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the correlation matrix using a [`GeneralWeightedCovariance`](@ref) estimator.

This method dispatches to [`robust_cor`](@ref), using the specified covariance estimator and optional observation weights stored in `ce`. If no weights are provided, the unweighted correlation is computed; otherwise, the weighted correlation is used.

# Arguments

  - `ce`: Covariance estimator containing the method and optional weights.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `mean`: Optional mean vector to use for centering.
  - `kwargs...`: Additional keyword arguments passed to [`robust_cor`](@ref).

# Returns

  - Correlation matrix as computed by the estimator and optional weights.

# Related

  - [`robust_cor`](@ref)
  - [`cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
    if isnothing(ce.w)
        robust_cor(ce.ce, X; dims = dims, mean = mean, kwargs...)
    else
        robust_cor(ce.ce, X, ce.w; dims = dims, mean = mean, kwargs...)
    end
end
function factory(ce::GeneralWeightedCovariance,
                 w::Union{Nothing, <:AbstractWeights} = nothing)
    return GeneralWeightedCovariance(; ce = ce.ce, w = isnothing(w) ? ce.w : w)
end

"""
    struct Covariance{T1, T2, T3} <: AbstractCovarianceEstimator
        me::T1
        ce::T2
        alg::T3
    end

A flexible container type for configuring and applying joint expected returns and covariance estimation in PortfolioOptimisers.jl.

`Covariance` encapsulates all components required for estimating the mean vector and covariance matrix of asset returns, including the expected returns estimator, the covariance estimator, and the moment algorithm. This enables modular and extensible workflows for portfolio optimization and risk modeling.

# Fields

  - `me`: Expected returns estimator.
  - `ce`: Covariance estimator.
  - `alg`: Moment algorithm.

# Constructor

    Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                 ce::StatsBase.CovarianceEstimator = GeneralWeightedCovariance(),
                 alg::AbstractMomentAlgorithm = Full())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> cov_est = Covariance()
Covariance
   me | SimpleExpectedReturns
      |   w | nothing
   ce | GeneralWeightedCovariance
      |   ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
      |    w | nothing
  alg | Full()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralWeightedCovariance`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
"""
struct Covariance{T1, T2, T3} <: AbstractCovarianceEstimator
    me::T1
    ce::T2
    alg::T3
end
function Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    ce::StatsBase.CovarianceEstimator = GeneralWeightedCovariance(),
                    alg::AbstractMomentAlgorithm = Full())
    return Covariance(me, ce, alg)
end
function factory(ce::Covariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Covariance(; me = factory(ce.me, w), ce = factory(ce.ce, w), alg = ce.alg)
end

"""
    cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the full covariance matrix using a [`Covariance`](@ref) estimator.

# Arguments

  - `ce`: Covariance estimator with `Full` moment algorithm.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the covariance.
  - `mean`: Optional mean vector for centering. If not provided, computed using `ce.me`.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

# Returns

  - Covariance matrix as computed by the estimator and moment algorithm.

# Related

  - [`Covariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralWeightedCovariance`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
  - [`cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`cor(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`cor(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    return cov(ce.ce, X; dims = dims, mean = mu, kwargs...)
end
"""
    cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the semi covariance matrix using a [`Covariance`](@ref) estimator.

# Arguments

  - `ce`: Covariance estimator with `Semi` moment algorithm.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the covariance.
  - `mean`: Optional mean vector for centering. If not provided, computed using `ce.me`.
  - `kwargs...`: Additional keyword arguments passed to the underlying covariance estimator.

# Returns

  - Covariance matrix as computed by the estimator and moment algorithm.

# Related

  - [`Covariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralWeightedCovariance`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
  - [`cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`cor(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`cor(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return cov(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end

"""
    cor(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the full correlation matrix using a [`Covariance`](@ref) estimator.

# Arguments

  - `ce`: Covariance estimator with `Full` moment algorithm.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `mean`: Optional mean vector for centering. If not provided, computed using `ce.me`.
  - `kwargs...`: Additional keyword arguments passed to the underlying correlation estimator.

# Returns

  - Correlation matrix as computed by the estimator and moment algorithm.

# Related

  - [`Covariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralWeightedCovariance`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
  - [`cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`cor(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    return cor(ce.ce, X; dims = dims, mean = mu, kwargs...)
end
"""
    cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the semi correlation matrix using a [`Covariance`](@ref) estimator.

# Arguments

  - `ce`: Covariance estimator with `Semi` moment algorithm.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the correlation.
  - `mean`: Optional mean vector for centering. If not provided, computed using `ce.me`.
  - `kwargs...`: Additional keyword arguments passed to the underlying correlation estimator.

# Returns

  - Correlation matrix as computed by the estimator and moment algorithm.

# Related

  - [`Covariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralWeightedCovariance`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
  - [`cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`cor(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return cor(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end

export GeneralWeightedCovariance, Covariance
