"""
    abstract type CoskewnessEstimator <: AbstractEstimator end

Abstract supertype for all coskewness estimators in PortfolioOptimisers.jl.

All concrete types implementing coskewness estimation algorithms should subtype `CoskewnessEstimator`. This enables a consistent interface for coskewness-based higher moment estimators throughout the package.

# Related

  - [`Coskewness`](@ref)
  - [`AbstractEstimator`](@ref)
"""
abstract type CoskewnessEstimator <: AbstractEstimator end

"""
    struct Coskewness{T1 <: AbstractExpectedReturnsEstimator,
                      T2 <: AbstractMatrixProcessingEstimator,
                      T3 <: AbstractMomentAlgorithm} <: CoskewnessEstimator
        me::T1
        mp::T2
        alg::T3
    end

Container type for coskewness estimators.

`Coskewness` encapsulates the mean estimator, matrix processing estimator, and moment algorithm for coskewness estimation. This enables modular workflows for higher-moment portfolio analysis.

# Fields

  - `me::AbstractExpectedReturnsEstimator`: Mean estimator for expected returns.
  - `mp::AbstractMatrixProcessingEstimator`: Matrix processing estimator for coskewness tensors.
  - `alg::AbstractMomentAlgorithm`: Moment algorithm (e.g., `Full`, `Semi`).

# Constructor

    Coskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                mp::AbstractMatrixProcessingEstimator = NonPositiveDefiniteMatrixProcessing(),
                alg::AbstractMomentAlgorithm = Full())

Construct a `Coskewness` estimator with the specified mean estimator, matrix processing estimator, and moment algorithm.

# Related

  - [`CoskewnessEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
"""
struct Coskewness{T1 <: AbstractExpectedReturnsEstimator,
                  T2 <: AbstractMatrixProcessingEstimator, T3 <: AbstractMomentAlgorithm} <:
       CoskewnessEstimator
    me::T1
    mp::T2
    alg::T3
end
"""
    Coskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                mp::AbstractMatrixProcessingEstimator = NonPositiveDefiniteMatrixProcessing(),
                alg::AbstractMomentAlgorithm = Full())

Construct a [`Coskewness`](@ref) estimator for coskewness computation.

# Arguments

  - `me::AbstractExpectedReturnsEstimator`: Mean estimator for expected returns.
  - `mp::AbstractMatrixProcessingEstimator`: Matrix processing estimator.
  - `alg::AbstractMomentAlgorithm`: Moment algorithm.

# Returns

  - `Coskewness`: Configured coskewness estimator.

# Examples

```jldoctest
julia> Coskewness()
Coskewness
   me | SimpleExpectedReturns
      |   w | nothing
   mp | NonPositiveDefiniteMatrixProcessing
      |   denoise | nothing
      |    detone | nothing
      |       alg | nothing
  alg | Full()
```

# Related

  - [`CoskewnessEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
"""
function Coskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = NonPositiveDefiniteMatrixProcessing(),
                    alg::AbstractMomentAlgorithm = Full())
    return Coskewness{typeof(me), typeof(mp), typeof(alg)}(me, mp, alg)
end
function factory(ce::Coskewness, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Coskewness(; me = factory(ce.me, w), mp = ce.mp, alg = ce.alg)
end

"""
    __coskewness(cskew::AbstractMatrix, X::AbstractMatrix, mp::AbstractMatrixProcessingEstimator)

Internal helper for coskewness matrix processing.

`__coskewness` processes the coskewness tensor by applying the matrix processing estimator to each block, then projects the result using eigenvalue decomposition and clamps negative values. Used internally for robust coskewness estimation.

# Arguments

  - `cskew::AbstractMatrix`: Coskewness tensor (flattened or block matrix).
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `mp::AbstractMatrixProcessingEstimator`: Matrix processing estimator.

# Returns

  - `V::Matrix`: Processed coskewness matrix.

# Related

  - [`Coskewness`](@ref)
  - [`_coskewness`](@ref)
  - [`matrix_processing!`](@ref)
  - [`coskewness`](@ref)
"""
function __coskewness(cskew::AbstractMatrix, X::AbstractMatrix,
                      mp::AbstractMatrixProcessingEstimator)
    N = size(cskew, 1)
    V = zeros(eltype(cskew), N, N)
    for i in 1:N
        j = (i - 1) * N + 1
        k = i * N
        coskew_jk = view(cskew, :, j:k)
        matrix_processing!(mp, coskew_jk, X)
        vals, vecs = eigen(coskew_jk)
        vals .= clamp.(real.(vals), typemin(eltype(cskew)), zero(eltype(cskew))) +
                clamp.(imag.(vals), typemin(eltype(cskew)), zero(eltype(cskew)))im
        V .-= real(vecs * Diagonal(vals) * transpose(vecs))
    end
    return V
end

"""
    _coskewness(y::AbstractMatrix, X::AbstractMatrix, mp::AbstractMatrixProcessingEstimator)

Internal helper for coskewness computation.

`_coskewness` computes the coskewness tensor and applies matrix processing. Used internally by coskewness estimators.

# Arguments

  - `y::AbstractMatrix`: Centered data vector (e.g., `X .- mean`).
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `mp::AbstractMatrixProcessingEstimator`: Matrix processing estimator.

# Returns

  - `cskew::Matrix`: Coskewness tensor.
  - `V::Matrix`: Processed coskewness matrix.

# Related

  - [`Coskewness`](@ref)
  - [`__coskewness`](@ref)
  - [`coskewness`](@ref)
"""
function _coskewness(y::AbstractMatrix, X::AbstractMatrix,
                     mp::AbstractMatrixProcessingEstimator)
    o = transpose(range(; start = one(eltype(y)), stop = one(eltype(y)),
                        length = size(X, 2)))
    z = kron(o, y) ⊙ kron(y, o)
    cskew = transpose(X) * z / size(X, 1)
    V = __coskewness(cskew, X, mp)
    return cskew, V
end
"""
    coskewness(ske::Coskewness{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
    coskewness(ske::Coskewness{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
    coskewness(::Nothing, args...; kwargs...)

Compute the full coskewness tensor and processed matrix for a dataset. For `Full`, it uses all centered data; for `Semi`, it uses only negative deviations. If the estimator is `nothing`, returns `(nothing, nothing)`.

# Arguments

  - `ske::Coskewness{<:Any, <:Any, <:Full}`: Coskewness estimator with `Full` moment algorithm.
  - `ske::Coskewness{<:Any, <:Any, <:Semi}`: Coskewness estimator with `Semi` moment algorithm.
  - `ske::Nothing`: No-op coskewness computation, returns `(nothing, nothing)`.
  - `X::AbstractMatrix`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the mean.
  - `mean`: Optional mean vector. If not provided, computed using the estimator's mean estimator.
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - `cskew::Matrix`: Coskewness tensor (observations × assets^2).
  - `V::Matrix`: Processed coskewness matrix (assets × assets).

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = randn(rng, 10, 3);

julia> cskew, V = coskewness(Coskewness(), X);

julia> cskew
3×9 Matrix{Float64}:
 -0.456556   0.104588   0.391789  …   0.391789  -0.283963   0.025956
 -0.136453  -0.191539  -0.139315     -0.139315   0.210037  -0.0952308
  0.176565  -0.219895   0.24526       0.24526    0.105632  -0.772302

julia> V
3×3 Matrix{Float64}:
 0.74159    0.428314   0.0676831
 0.428314   0.316494   0.0754933
 0.0676831  0.0754933  0.833249
```

# Related

  - [`Coskewness`](@ref)
  - [`_coskewness`](@ref)
  - [`__coskewness`](@ref)
"""
function coskewness(ske::Coskewness{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? Statistics.mean(ske.me, X; kwargs...) : mean
    y = X .- mu
    return _coskewness(y, X, ske.mp)
end
function coskewness(ske::Coskewness{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing, kwargs...)
    @smart_assert(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? Statistics.mean(ske.me, X; kwargs...) : mean
    y = min.(X .- mu, zero(eltype(X)))
    return _coskewness(y, X, ske.mp)
end
function coskewness(::Nothing, args...; kwargs...)
    return nothing, nothing
end

export Coskewness, coskewness
