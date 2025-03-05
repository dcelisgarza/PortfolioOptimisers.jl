struct FullCoskewness{T1 <: ExpectedReturnsEstimator, T2 <: MatrixProcessing} <:
       CoskewnessEstimator
    me::T1
    mp::T2
end
function FullCoskewness(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::MatrixProcessing = NonPositiveDefiniteMatrixProcessing())
    return FullCoskewness{typeof(me), typeof(mp)}(me, mp)
end
function coskewness(ske::FullCoskewness, X::AbstractMatrix; dims::Int = 1, kwargs...)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = mean(ske.me, X; kwargs...)
    y = X .- mu
    return _coskewness(y, X, ske.mp)
end

export FullCoskewness
