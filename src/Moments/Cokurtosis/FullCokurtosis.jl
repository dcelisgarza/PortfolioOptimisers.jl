struct FullCokurtosis{T1 <: AbstractExpectedReturnsEstimator,
                      T2 <: AbstractMatrixProcessingEstimator} <: CokurtosisEstimator
    me::T1
    mp::T2
end
function FullCokurtosis(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::AbstractMatrixProcessingEstimator = DefaultMatrixProcessing())
    return FullCokurtosis{typeof(me), typeof(mp)}(me, mp)
end
function cokurtosis(ke::FullCokurtosis, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = mean(ke.me, X)
    X = X .- mu
    return _cokurosis(X, ke.mp)
end
function factory(ce::FullCokurtosis, w::Union{Nothing, <:AbstractWeights} = nothing)
    return FullCokurtosis(; me = factory(ce.me, w), mp = ce.mp)
end

export FullCokurtosis
