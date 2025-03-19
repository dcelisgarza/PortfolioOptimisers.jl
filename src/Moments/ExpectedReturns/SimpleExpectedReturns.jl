struct SimpleExpectedReturns{T1 <: Union{Nothing, <:AbstractWeights}} <:
       ExpectedReturnsEstimator
    w::T1
end
function SimpleExpectedReturns(; w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleExpectedReturns(w)
end
function StatsBase.mean(me::SimpleExpectedReturns, X::AbstractMatrix; dims::Int = 1)
    return isnothing(me.w) ? mean(X; dims = dims) : mean(X, me.w; dims = dims)
end
function moment_factory_w(::SimpleExpectedReturns,
                          w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleExpectedReturns(; w = w)
end

export SimpleExpectedReturns
