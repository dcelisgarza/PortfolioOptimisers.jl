struct SimpleExpectedReturns{T1 <: Union{Nothing, <:AbstractWeights}} <:
       AbstractExpectedReturnsEstimator
    w::T1
end
function SimpleExpectedReturns(; w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleExpectedReturns(w)
end
function StatsBase.mean(me::SimpleExpectedReturns, X::AbstractArray; dims::Int = 1,
                        kwargs...)
    return isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims)
end
function factory(me::SimpleExpectedReturns, w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleExpectedReturns(; w = isnothing(w) ? me.w : w)
end

export SimpleExpectedReturns
