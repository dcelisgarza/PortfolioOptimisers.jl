struct ExcessExpectedReturns{T1 <: AbstractExpectedReturnsEstimator, T2 <: Real} <:
       AbstractShrunkExpectedReturnsEstimator
    me::T1
    rf::T2
end
function ExcessExpectedReturns(;
                               me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                               rf::Real = 0.0)
    return ExcessExpectedReturns{typeof(me), typeof(rf)}(me, rf)
end
function StatsBase.mean(me::ExcessExpectedReturns, X::AbstractArray; dims::Int = 1,
                        kwargs...)
    return mean(me.me, X; dims = dims, kwargs...) .- me.rf
end
function factory(me::ExcessExpectedReturns, w::Union{Nothing, <:AbstractWeights} = nothing)
    return ExcessExpectedReturns(; me = factory(me.me, w), rf = me.rf)
end

export ExcessExpectedReturns
