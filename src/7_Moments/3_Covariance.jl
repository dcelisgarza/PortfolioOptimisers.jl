"""
    GeneralWeightedCovariance{T1 <: StatsBase.CovarianceEstimator,
                              T2 <: Union{Nothing, <:AbstractWeights}} <: AbstractCovarianceEstimator

A flexible covariance estimator for PortfolioOptimisers.jl supporting arbitrary covariance estimators and optional observation weights.

`GeneralWeightedCovariance` allows users to specify both the covariance estimation method and optional observation weights. This enables robust and extensible covariance estimation workflows.

# Fields

  - `ce::StatsBase.CovarianceEstimator`: Covariance estimator.
  - `w::Union{Nothing, <:AbstractWeights}`: Optional weights for each observation. If `nothing`, the estimator is unweighted.

# Constructor

    GeneralWeightedCovariance(; ce::StatsBase.CovarianceEstimator = SimpleCovariance(; corrected = true),
                               w::Union{Nothing, <:AbstractWeights} = nothing)

Construct a `GeneralWeightedCovariance` estimator with the specified covariance estimator and optional weights.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
struct GeneralWeightedCovariance{T1 <: StatsBase.CovarianceEstimator,
                                 T2 <: Union{Nothing, <:AbstractWeights}} <:
       AbstractCovarianceEstimator
    ce::T1
    w::T2
end

"""
    GeneralWeightedCovariance(; ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true),
                               w::Union{Nothing, <:AbstractWeights} = nothing)

Construct a [`GeneralWeightedCovariance`](@ref) estimator for flexible covariance estimation with optional observation weights.

This constructor creates a `GeneralWeightedCovariance` object using the specified covariance estimator and optional weights. If no weights are provided, the estimator defaults to unweighted covariance estimation. If weights are provided, they must not be empty.

# Arguments

  - `ce::StatsBase.CovarianceEstimator`: Covariance estimator to use.
  - `w::Union{Nothing, <:AbstractWeights}`: Optional observation weights. If `nothing`, the estimator is unweighted. If provided, must be non-empty.

# Returns

  - `GeneralWeightedCovariance`: A covariance estimator configured with the specified method and optional weights.

# Validation

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

  - [`GeneralWeightedCovariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function GeneralWeightedCovariance(;
                                   ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(;
                                                                                                  corrected = true),
                                   w::Union{Nothing, <:AbstractWeights} = nothing)
    if isa(w, AbstractWeights)
        @smart_assert(!isempty(w))
    end
    return GeneralWeightedCovariance{typeof(ce), typeof(w)}(ce, w)
end

"""
    cov(ce::GeneralWeightedCovariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the covariance matrix using a [`GeneralWeightedCovariance`](@ref) estimator.

This method dispatches to [`robust_cov`](@ref), using the specified covariance estimator and optional observation weights stored in `ce`. If no weights are provided, the unweighted covariance is computed; otherwise, the weighted covariance is used.

# Arguments

  - `ce::GeneralWeightedCovariance`: Covariance estimator containing the method and optional weights.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
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

  - `ce::GeneralWeightedCovariance`: Covariance estimator containing the method and optional weights.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
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
    struct Covariance{T1 <: AbstractExpectedReturnsEstimator,
                      T2 <: StatsBase.CovarianceEstimator,
                      T3 <: AbstractMomentAlgorithm} <: AbstractCovarianceEstimator
        me::T1
        ce::T2
        alg::T3
    end

A flexible container type for configuring and applying joint expected returns and covariance estimation in PortfolioOptimisers.jl.

`Covariance` encapsulates all components required for estimating the mean vector and covariance matrix of asset returns, including the expected returns estimator, the covariance estimator, and the moment algorithm. This enables modular and extensible workflows for portfolio optimization and risk modeling.

# Fields

  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator.
  - `ce::StatsBase.CovarianceEstimator`: Covariance estimator.
  - `alg::AbstractMomentAlgorithm`: Moment algorithm.

# Constructor

    Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                ce::StatsBase.CovarianceEstimator = GeneralWeightedCovariance(),
                alg::AbstractMomentAlgorithm = Full())

Construct a `Covariance` estimator with the specified expected returns estimator, covariance estimator, and moment algorithm.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralWeightedCovariance`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
"""
struct Covariance{T1 <: AbstractExpectedReturnsEstimator,
                  T2 <: StatsBase.CovarianceEstimator, T3 <: AbstractMomentAlgorithm} <:
       AbstractCovarianceEstimator
    me::T1
    ce::T2
    alg::T3
end
"""
    Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                ce::StatsBase.CovarianceEstimator = GeneralWeightedCovariance(),
                alg::AbstractMomentAlgorithm = Full())

Construct a [`Covariance`](@ref) estimator for joint mean and covariance estimation.

This constructor creates a `Covariance` object using the specified expected returns estimator, covariance estimator, and moment algorithm. Defaults are provided for each component to enable robust and extensible estimation workflows.

# Arguments

  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator.
  - `ce::StatsBase.CovarianceEstimator`: Covariance estimator.
  - `alg::AbstractMomentAlgorithm`: Moment algorithm.

# Returns

  - `Covariance`: A configured joint mean and covariance estimator.

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

  - [`Covariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
  - [`GeneralWeightedCovariance`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`Full`](@ref)
  - [`Semi`](@ref)
"""
function Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    ce::StatsBase.CovarianceEstimator = GeneralWeightedCovariance(),
                    alg::AbstractMomentAlgorithm = Full())
    return Covariance{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
end
function factory(ce::Covariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Covariance(; me = factory(ce.me, w), ce = factory(ce.ce, w), alg = ce.alg)
end

"""
    cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the full covariance matrix using a [`Covariance`](@ref) estimator.

# Arguments

  - `ce::Covariance{<:Any, <:Any, <:Full}`: Covariance estimator with `Full` moment algorithm.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the covariance.
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

  - `ce::Covariance{<:Any, <:Any, <:Semi}`: Covariance estimator with `Semi` moment algorithm.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the covariance.
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

  - `ce::Covariance{<:Any, <:Any, <:Full}`: Covariance estimator with `Full` moment algorithm.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the correlation.
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

  - `ce::Covariance{<:Any, <:Any, <:Semi}`: Covariance estimator with `Semi` moment algorithm.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the correlation.
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
