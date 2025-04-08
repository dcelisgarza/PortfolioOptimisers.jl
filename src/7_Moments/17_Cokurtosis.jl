abstract type CokurtosisEstimator end
function cokurtosis(::Nothing, args...; kwargs...)
    return nothing
end
struct Cokurtosis{T1 <: AbstractMomentAlgorithm, T2 <: AbstractExpectedReturnsEstimator,
                  T3 <: AbstractMatrixProcessingEstimator} <: CokurtosisEstimator
    alg::T1
    me::T2
    mp::T3
end
function Cokurtosis(; alg::AbstractMomentAlgorithm = Full(),
                    me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                    mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing())
    return Cokurtosis{typeof(alg), typeof(me), typeof(mp)}(alg, me, mp)
end
function factory(ce::Cokurtosis, w::Union{Nothing, <:AbstractWeights} = nothing)
    return Cokurtosis(; alg = ce.alg, me = factory(ce.me, w), mp = ce.mp)
end
function _cokurosis(X, mp)
    T, N = size(X)
    o = transpose(range(; start = one(eltype(X)), stop = one(eltype(X)), length = N))
    z = kron(o, X) .* kron(X, o)
    ckurt = transpose(z) * z / T
    matrix_processing!(mp, ckurt, X)
    return ckurt
end
function cokurtosis(ke::Cokurtosis{<:Full, <:Any, <:Any}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? StatsBase.mean(ke.me, X) : mean
    X = X .- mu
    return _cokurosis(X, ke.mp)
end
function cokurtosis(ke::Cokurtosis{<:Semi, <:Any, <:Any}, X::AbstractMatrix; dims::Int = 1,
                    mean = nothing)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = isnothing(mean) ? StatsBase.mean(ke.me, X) : mean
    X = min.(X .- mu, zero(eltype(X)))
    return _cokurosis(X, ke.mp)
end

export cokurtosis, Cokurtosis
