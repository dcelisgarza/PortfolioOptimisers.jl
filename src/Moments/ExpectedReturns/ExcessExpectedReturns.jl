struct ExcessExpectedReturns{T1 <: ExpectedReturnsEstimator, T2 <: Real} <:
       ShrunkExpectedReturnsEstimator
    me::T1
    rf::T2
end
function ExcessExpectedReturns(; me::ExpectedReturnsEstimator = SimpleExpectedReturns(),
                               rf::Real = 0.0)
    return ExcessExpectedReturns{typeof(me), typeof(rf)}(me, rf)
end
function StatsBase.mean(me::ExcessExpectedReturns, X::AbstractMatrix; dims::Int = 1)
    return mean(me.me, X; dims = dims) .- me.rf
end

export ExcessExpectedReturns
