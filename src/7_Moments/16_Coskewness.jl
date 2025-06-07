abstract type CoskewnessEstimator end
function coskewness(::Nothing, args...; kwargs...)
    return nothing, nothing
end
struct Coskewness{T1 <: AbstractExpectedReturnsEstimator,
                  T2 <: AbstractMatrixProcessingEstimator, T3 <: AbstractMomentAlgorithm} <:
       CoskewnessEstimator
    me::T1
    mp::T2
    alg::T3
end
function Coskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = NonPositiveDefiniteMatrixProcessing(),
                    alg::AbstractMomentAlgorithm = Full())
    return Coskewness{typeof(me), typeof(mp), typeof(alg)}(me, mp, alg)
end
function factory(ce::Coskewness, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Coskewness(; me = factory(ce.me, w), mp = ce.mp, alg = ce.alg)
end
function __coskewness(cskew, X, mp)
    N = size(cskew, 1)
    V = zeros(eltype(cskew), N, N)
    for i ∈ 1:N
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
function _coskewness(y, X, mp)
    o = transpose(range(; start = one(eltype(y)), stop = one(eltype(y)),
                        length = size(X, 2)))
    z = kron(o, y) ⊙ kron(y, o)
    cskew = transpose(X) * z / size(X, 1)
    V = __coskewness(cskew, X, mp)
    return cskew, V
end
function coskewness(ske::Coskewness{<:Any, <:Any, <:Full}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? StatsBase.mean(ske.me, X) : mean
    y = X .- mu
    return _coskewness(y, X, ske.mp)
end
function coskewness(ske::Coskewness{<:Any, <:Any, <:Semi}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? StatsBase.mean(ske.me, X) : mean
    y = min.(X .- mu, zero(eltype(X)))
    return _coskewness(y, X, ske.mp)
end

export Coskewness, coskewness
