"""
    GeneralWeightedCovariance{T1 <: StatsBase.CovarianceEstimator,
                              T2 <: Union{Nothing, <:AbstractWeights}} <: AbstractCovarianceEstimator

A flexible covariance estimator for PortfolioOptimisers.jl supporting arbitrary covariance estimators and optional observation weights.

`GeneralWeightedCovariance` allows users to specify both the covariance estimation method (e.g., sample, shrinkage) and optional observation weights. This enables robust and extensible covariance estimation workflows.

# Fields

  - `ce::StatsBase.CovarianceEstimator`: Covariance estimator (e.g., `SimpleCovariance`, `LedoitWolfCovariance`).
  - `w::Union{Nothing, <:AbstractWeights}`: Optional weights for each observation. If `nothing`, the estimator is unweighted.

# Constructor

    GeneralWeightedCovariance(; ce::StatsBase.CovarianceEstimator = SimpleCovariance(; corrected = true),
                               w::Union{Nothing, <:AbstractWeights} = nothing)

Construct a `GeneralWeightedCovariance` estimator with the specified covariance estimator and optional weights.

# Related

  - [`AbstractCovarianceEstimator`](@ref)
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

  - `ce::StatsBase.CovarianceEstimator = StatsBase.SimpleCovariance(; corrected = true)`: Covariance estimator to use.
  - `w::Union{Nothing, <:AbstractWeights} = nothing`: Optional observation weights. If `nothing`, the estimator is unweighted. If provided, must be non-empty.

# Returns

  - `GeneralWeightedCovariance`: A covariance estimator configured with the specified method and optional weights.

# Validation

  - If `w` is provided, it must not be empty.

# Examples

```jldoctest
julia> using StatsBase

julia> gwc = GeneralWeightedCovariance()
GeneralWeightedCovariance
  ce | StatsBase.SimpleCovariance: SimpleCovariance(true)
   w | nothing

julia> w = Weights([0.1, 0.2, 0.7]);

julia> gwc = GeneralWeightedCovariance(; w = w)
GeneralWeightedCovariance
  ce | StatsBase.SimpleCovariance: SimpleCovariance(true)
   w | StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.1, 0.2, 0.7]
```

# Related

  - [`GeneralWeightedCovariance`](@ref)
  - [`AbstractCovarianceEstimator`](@ref)
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
function Base.show(io::IO, gwc::GeneralWeightedCovariance)
    println(io, "GeneralWeightedCovariance")
    for field in fieldnames(typeof(gwc))
        val = getfield(gwc, field)
        print(io, "  ", lpad(string(field), 2), " ")
        if isnothing(val)
            println(io, "| nothing")
        else
            println(io, "| $(typeof(val)): ", repr(val))
        end
    end
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
struct Covariance{T1 <: AbstractExpectedReturnsEstimator,
                  T2 <: StatsBase.CovarianceEstimator, T3 <: AbstractMomentAlgorithm} <:
       AbstractCovarianceEstimator
    me::T1
    ce::T2
    alg::T3
end
function Covariance(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    ce::StatsBase.CovarianceEstimator = GeneralWeightedCovariance(),
                    alg::AbstractMomentAlgorithm = Full())
    return Covariance{typeof(me), typeof(ce), typeof(alg)}(me, ce, alg)
end
function Base.show(io::IO, cov::Covariance)
    println(io, "Covariance")
    for field in fieldnames(typeof(cov))
        val = getfield(cov, field)
        print(io, "  ", lpad(string(field), 3), " ")
        if isnothing(val)
            println(io, "| nothing")
        elseif isa(val, AbstractExpectedReturnsEstimator) ||
               isa(val, AbstractCovarianceEstimator) ||
               isa(val, AbstractMomentAlgorithm)
            ioalg = IOBuffer()
            show(ioalg, val)
            algstr = String(take!(ioalg))
            alglines = split(algstr, '\n')
            println(io, "| ", alglines[1])
            for l in alglines[2:end]
                println(io, "      | ", l)
            end
        else
            println(io, "| $(typeof(val)): ", repr(val))
        end
    end
end
function factory(ce::Covariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Covariance(; me = factory(ce.me, w), ce = factory(ce.ce, w), alg = ce.alg)
end
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    return cov(ce.ce, X; dims = dims, mean = mu, kwargs...)
end
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:Full}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    return cor(ce.ce, X; dims = dims, mean = mu, kwargs...)
end
function Statistics.cov(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return cov(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end
function Statistics.cor(ce::Covariance{<:Any, <:Any, <:Semi}, X::AbstractMatrix;
                        dims::Int = 1, mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ce.me, X; dims = dims, kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return cor(ce.ce, X; dims = dims, mean = zero(eltype(X)), kwargs...)
end

export GeneralWeightedCovariance, Covariance
