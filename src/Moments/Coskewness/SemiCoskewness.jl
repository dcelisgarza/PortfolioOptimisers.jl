struct SemiCoskewness{T1 <: ExpectedReturnsEstimator, T2 <: MatrixProcessing} <:
       CoskewnessEstimator
    me::T1
    mp::T2
end
function SemiCoskewness(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                        mp::MatrixProcessing = NonPositiveDefiniteMatrixProcessing())
    return SemiCoskewness{typeof(me), typeof(mp)}(me, mp)
end
function coskewness(ske::SemiCoskewness, X::AbstractMatrix; dims::Int = 1)
    @smart_assert(dims ∈ (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = mean(ske.me, X)
    y = min.(X .- mu, zero(eltype(X)))
    return _coskewness(y, X, ske.mp)
end
function moment_factory_w(ce::SemiCoskewness,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return SemiCoskewness(; me = moment_factory_w(ce.me, w), mp = ce.mp)
end

export SemiCoskewness
