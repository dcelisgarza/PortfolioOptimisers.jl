struct FullCoskewness{T1 <: ExpectedReturnsEstimator, T2 <: MatrixProcessing} <:
       CoskewnessEstimator
    me::T1
    mp::T2
end
function FullCoskewness(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::MatrixProcessing = NonPositiveDefiniteMatrixProcessing())
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
function moment_factory_w(ce::FullCoskewness,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return FullCoskewness(; me = moment_factory_w(ce.me, w), mp = ce.mp)
end

export FullCoskewness
