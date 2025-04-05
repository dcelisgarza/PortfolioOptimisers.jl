struct FullCoskewness{T1 <: AbstractExpectedReturnsEstimator,
                      T2 <: AbstractMatrixProcessingEstimator} <: CoskewnessEstimator
    me::T1
    mp::T2
end
function FullCoskewness(; me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::AbstractMatrixProcessingEstimator = NonPositiveDefiniteMatrixProcessing())
    return FullCoskewness{typeof(me), typeof(mp)}(me, mp)
end
function coskewness(ske::FullCoskewness, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = mean(ske.me, X)
    y = X .- mu
    return _coskewness(y, X, ske.mp)
end
function w_moment_factory(ce::FullCoskewness,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return FullCoskewness(; me = w_moment_factory(ce.me, w), mp = ce.mp)
end

export FullCoskewness
