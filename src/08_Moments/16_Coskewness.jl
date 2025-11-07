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
    struct Coskewness{T1, T2, T3} <: CoskewnessEstimator
        me::T1
        mp::T2
        alg::T3
    end

Container type for coskewness estimators.

`Coskewness` encapsulates the mean estimator, matrix processing estimator, and moment algorithm for coskewness estimation. This enables modular workflows for higher-moment portfolio analysis.

# Fields

  - `me`: Mean estimator for expected returns.
  - `mp`: Matrix processing estimator for coskewness tensors.
  - `alg`: Moment algorithm.

# Constructor

    Coskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
               mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
               alg::AbstractMomentAlgorithm = Full())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> Coskewness()
Coskewness
   me ┼ SimpleExpectedReturns
      │   w ┴ nothing
   mp ┼ DefaultMatrixProcessing
      │       pdm ┼ Posdef
      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
      │   denoise ┼ nothing
      │    detone ┼ nothing
      │       alg ┴ nothing
  alg ┴ Full()
```

# Related

  - [`CoskewnessEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractMatrixProcessingEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
"""
struct Coskewness{T1, T2, T3} <: CoskewnessEstimator
    me::T1
    mp::T2
    alg::T3
    function Coskewness(me::AbstractExpectedReturnsEstimator,
                        mp::AbstractMatrixProcessingEstimator, alg::AbstractMomentAlgorithm)
        return new{typeof(me), typeof(mp), typeof(alg)}(me, mp, alg)
    end
end
function Coskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing(),
                    alg::AbstractMomentAlgorithm = Full())
    return Coskewness(me, mp, alg)
end
function factory(ce::Coskewness, w::WeightsType = nothing)
    return Coskewness(; me = factory(ce.me, w), mp = ce.mp, alg = ce.alg)
end
"""
    __coskewness(cskew::NumMat, X::NumMat,
                 mp::AbstractMatrixProcessingEstimator)

Internal helper for coskewness matrix processing.

`__coskewness` processes the coskewness tensor by applying the matrix processing estimator to each block, then projects the result using eigenvalue decomposition and clamps negative values. Used internally for robust coskewness estimation.

# Arguments

  - `cskew`: Coskewness tensor (flattened or block matrix).
  - `X`: Data matrix (observations × assets).
  - `mp`: Matrix processing estimator.

# Returns

  - `V::Matrix{<:Number}`: Processed coskewness matrix.

# Related

  - [`Coskewness`](@ref)
  - [`_coskewness`](@ref)
  - [`matrix_processing!`](@ref)
  - [`coskewness`](@ref)
"""
function __coskewness(cskew::NumMat, X::NumMat, mp::AbstractMatrixProcessingEstimator)
    N = size(cskew, 1)
    V = zeros(eltype(cskew), N, N)
    for i in 1:N
        j = (i - 1) * N + 1
        k = i * N
        coskew_jk = view(cskew, :, j:k)
        vals, vecs = eigen(coskew_jk)
        if isa(eltype(vals), Number)
            vals .= clamp.(vals, typemin(eltype(cskew)), zero(eltype(cskew)))
            V .-= vecs * Diagonal(vals) * transpose(vecs)
        else
            vals .= clamp.(real.(vals), typemin(eltype(cskew)), zero(eltype(cskew))) +
                    clamp.(imag.(vals), typemin(eltype(cskew)), zero(eltype(cskew)))im
            V .-= real(vecs * Diagonal(vals) * transpose(vecs))
        end
    end
    matrix_processing!(mp, V, X)
    return V
end
"""
    _coskewness(Y::NumMat, X::NumMat, mp::AbstractMatrixProcessingEstimator)

Internal helper for coskewness computation.

`_coskewness` computes the coskewness tensor and applies matrix processing. Used internally by coskewness estimators.

# Arguments

  - `Y`: Centered data vector (e.g., `X .- mean`).
  - `X`: Data matrix (observations × assets).
  - `mp`: Matrix processing estimator.

# Returns

  - `cskew::Matrix{<:Number}`: Coskewness tensor.
  - `V::Matrix{<:Number}`: Processed coskewness matrix.

# Related

  - [`Coskewness`](@ref)
  - [`__coskewness`](@ref)
  - [`coskewness`](@ref)
"""
function _coskewness(Y::NumMat, X::NumMat, mp::AbstractMatrixProcessingEstimator)
    o = transpose(range(one(eltype(Y)), one(eltype(Y)); length = size(Y, 2)))
    z = kron(o, Y) ⊙ kron(Y, o)
    cskew = transpose(Y) * z / size(Y, 1)
    V = __coskewness(cskew, X, mp)
    return cskew, V
end
"""
    coskewness(ske::Union{Nothing, <:Coskewness}, X::NumMat; dims::Int = 1,
               mean = nothing, kwargs...)

Compute the full coskewness tensor and processed matrix for a dataset. For `Full`, it uses all centered data; for `Semi`, it uses only negative deviations. If the estimator is `nothing`, returns `(nothing, nothing)`.

# Arguments

  - `ske`: Coskewness estimator.

      + `ske::Coskewness{<:Any, <:Any, <:Full}`: Coskewness estimator with [`Full`](@ref) moment algorithm.
      + `ske::Coskewness{<:Any, <:Any, <:Semi}`: Coskewness estimator with [`Semi`](@ref) moment algorithm.
      + `ske::Nothing`: No-op, returns `(nothing, nothing)`.

  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the mean.
  - `mean`: Optional mean vector. If not provided, computed using the estimator's mean estimator.
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Validation

  - `dims` is either `1` or `2`.

# Returns

  - `cskew::Matrix{<:Number}`: Coskewness tensor (observations × assets^2).
  - `V::Matrix{<:Number}`: Processed coskewness matrix (assets × assets).

# Examples

```jldoctest
julia> using StableRNGs

julia> rng = StableRNG(123456789);

julia> X = randn(rng, 10, 3);

julia> cskew, V = coskewness(Coskewness(), X);

julia> cskew
3×9 Matrix{Float64}:
 -0.329646    0.0782455   0.325842  …   0.325842  -0.250881   0.16769
  0.0782455  -0.236104   -0.250881     -0.250881   0.266005   0.144546
  0.325842   -0.250881    0.16769       0.16769    0.144546  -0.605589

julia> V
3×3 Matrix{Float64}:
  0.513743   -0.0452078  -0.290893
 -0.0452078   0.402765   -0.0372996
 -0.290893   -0.0372996   0.837701
```

# Related

  - [`Coskewness`](@ref)
  - [`_coskewness`](@ref)
  - [`__coskewness`](@ref)
"""
function coskewness(ske::Coskewness{<:Any, <:Any, <:Full}, X::NumMat; dims::Int = 1,
                    mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? Statistics.mean(ske.me, X; kwargs...) : mean
    Y = X .- mu
    return _coskewness(Y, X, ske.mp)
end
function coskewness(ske::Coskewness{<:Any, <:Any, <:Semi}, X::NumMat; dims::Int = 1,
                    mean = nothing, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? Statistics.mean(ske.me, X; kwargs...) : mean
    Y = min.(X .- mu, zero(eltype(X)))
    return _coskewness(Y, X, ske.mp)
end
function coskewness(::Nothing, args...; kwargs...)
    return nothing, nothing
end

export Coskewness, coskewness
