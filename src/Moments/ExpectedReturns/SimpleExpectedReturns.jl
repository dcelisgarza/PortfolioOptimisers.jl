struct SimpleExpectedReturns{T1 <: Union{Nothing, <:AbstractWeights}} <:
       ExpectedReturnsEstimator
    w::T1
end
function SimpleExpectedReturns(; w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleExpectedReturns(w)
end
function StatsBase.mean(me::SimpleExpectedReturns, X::AbstractMatrix; dims::Int = 1,
                        kwargs...)
    return if isnothing(me.w)
        mean(X; dims = dims, kwargs...)
    else
        mean(X, me.w; dims = dims, kwargs...)
    end
end

export SimpleExpectedReturns
