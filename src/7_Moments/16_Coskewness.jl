abstract type CoskewnessEstimator end
function coskewness(::Nothing, args...; kwargs...)
    return nothing, nothing
end
struct Coskewness{T1 <: AbstractMomentAlgorithm, T2 <: AbstractExpectedReturnsEstimator,
                  T3 <: AbstractMatrixProcessingEstimator} <: CoskewnessEstimator
    alg::T1
    me::T2
    mp::T3
end
function Coskewness(; alg::AbstractMomentAlgorithm = Full(),
                    me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = NonPositiveDefiniteMatrixProcessing())
    return Coskewness{typeof(alg), typeof(me), typeof(mp)}(alg, me, mp)
end
function factory(ce::Coskewness, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Coskewness(; alg = ce.alg, me = factory(ce.me, w), mp = ce.mp)
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
    z = kron(o, y) .* kron(y, o)
    cskew = transpose(X) * z / size(X, 1)
    V = __coskewness(cskew, X, mp)
    return cskew, V
end
function coskewness(ske::Coskewness{<:Full, <:Any, <:Any}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? StatsBase.mean(ske.me, X) : mean
    y = X .- mu
    return _coskewness(y, X, ske.mp)
end
function coskewness(ske::Coskewness{<:Semi, <:Any, <:Any}, X::AbstractMatrix; dims::Int = 1,
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
