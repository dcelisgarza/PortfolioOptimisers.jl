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
function moment_factory_w(me::ExcessExpectedReturns,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return ExcessExpectedReturns(; me = moment_factory_w(me.me, w), rf = me.rf)
end

export ExcessExpectedReturns
