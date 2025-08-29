"""
    abstract type CokurtosisEstimator <: AbstractEstimator end

Abstract supertype for all cokurtosis estimators in PortfolioOptimisers.jl.

All concrete types implementing cokurtosis estimation algorithms should subtype `CokurtosisEstimator`. This enables a consistent interface for cokurtosis-based higher moment estimators throughout the package.

# Related

  - [`Cokurtosis`](@ref)
  - [`AbstractEstimator`](@ref)
"""
abstract type CokurtosisEstimator <: AbstractEstimator end

"""
    struct Cokurtosis{T1, T2, T3} <: CokurtosisEstimator
        me::T1
        mp::T2
        alg::T3
    end

Container type for cokurtosis estimators.

`Cokurtosis` encapsulates the mean estimator, matrix processing estimator, and moment algorithm for cokurtosis estimation. This enables modular workflows for higher-moment portfolio analysis.

# Fields

  - `me`: Mean estimator for expected returns.
  - `mp`: Matrix processing estimator for cokurtosis tensors.
  - `alg`: Moment algorithm.

# Constructor

    Cokurtosis(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                alg::AbstractMomentAlgorithm = Full())

Construct a `Cokurtosis` estimator with the specified mean estimator, matrix processing estimator, and moment algorithm.

# Related

  - [`CokurtosisEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
"""
struct Cokurtosis{T1, T2, T3} <: CokurtosisEstimator
    me::T1
    mp::T2
    alg::T3
end
"""
    Cokurtosis(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                alg::AbstractMomentAlgorithm = Full())

Construct a [`Cokurtosis`](@ref) estimator for cokurtosis computation.

# Arguments

  - `me`: Mean estimator for expected returns.
  - `mp`: Matrix processing estimator.
  - `alg`: Moment algorithm.

# Returns

  - `Cokurtosis`: Configured cokurtosis estimator.

# Examples

```jldoctest
julia> Cokurtosis()
Cokurtosis
   me | SimpleExpectedReturns
      |   w | nothing
   mp | DefaultMatrixProcessing
      |       pdm | Posdef
      |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
      |   denoise | nothing
      |    detone | nothing
      |       alg | nothing
  alg | Full()
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
"""
function Cokurtosis(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                    alg::AbstractMomentAlgorithm = Full())
    return Cokurtosis(me, mp, alg)
end
function factory(ce::Cokurtosis, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Cokurtosis(; me = factory(ce.me, w), mp = ce.mp, alg = ce.alg)
end

"""
    _cokurtosis(X::AbstractMatrix, mp::AbstractMatrixProcessingEstimator)

Internal helper for cokurtosis computation.

`_cokurtosis` computes the cokurtosis tensor for the input data matrix and applies matrix processing using the specified estimator.

# Arguments

  - `X`: Data matrix (observations × assets).
  - `mp`: Matrix processing estimator.

# Returns

  - `ckurt::Matrix{<:Real}`: Cokurtosis tensor after matrix processing.

# Related

  - [`Cokurtosis`](@ref)
  - [`matrix_processing!`](@ref)
  - [`cokurtosis`](@ref)
"""
function _cokurtosis(X::AbstractMatrix, mp::AbstractMatrixProcessingEstimator)
    T, N = size(X)
    o = transpose(range(; start = one(eltype(X)), stop = one(eltype(X)), length = N))
    z = kron(o, X) ⊙ kron(X, o)
    ckurt = transpose(z) * z / T
    matrix_processing!(mp, ckurt, X)
    return ckurt
end

"""
    cokurtosis(ke::Union{Nothing, <:Cokurtosis}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)

Compute the cokurtosis tensor for a dataset.

This method computes the cokurtosis tensor using the estimator's mean and matrix processing algorithm. For `Full`, it uses all centered data; for `Semi`, it uses only negative deviations. If the estimator is `nothing`, returns `nothing`.

# Arguments

  - `ke::Cokurtosis{<:Any, <:Any, <:Full}`: Cokurtosis estimator with `Full` moment algorithm.
  - `ke::Cokurtosis{<:Any, <:Any, <:Semi}`: Cokurtosis estimator with `Semi` moment algorithm.
  - `ke::Nothing`: No-op cokurtosis computation, returns `nothing`.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the mean.
  - `mean`: Optional mean vector. If not provided, computed using the estimator's mean estimator.
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - `ckurt::Matrix{<:Real}`: Cokurtosis tensor (assets^2 × assets^2).

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = randn(rng, 10, 2);

julia> cokurtosis(Cokurtosis(), X)
4×4 Matrix{Float64}:
  1.33947   -0.246726  -0.246726   0.493008
 -0.246726   0.493008   0.493008  -0.201444
 -0.246726   0.493008   0.493008  -0.201444
  0.493008  -0.201444  -0.201444   0.300335
```

# Related

  - [`Cokurtosis`](@ref)
  - [`_cokurtosis`](@ref)
"""
function cokurtosis(ke::Cokurtosis{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? Statistics.mean(ke.me, X; kwargs...) : mean
    X = X .- mu
    return _cokurtosis(X, ke.mp)
end
function cokurtosis(ke::Cokurtosis{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? Statistics.mean(ke.me, X; kwargs...) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return _cokurtosis(X, ke.mp)
end
function cokurtosis(::Nothing, args...; kwargs...)
    return nothing
end

export cokurtosis, Cokurtosis
