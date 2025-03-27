struct ThirdCentralMoment{T1 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                          T2 <: Union{Nothing, <:AbstractWeights},
                          T3 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetNoOptimisationRiskMeasure
    target::T1
    w::T2
    mu::T3
end
function ThirdCentralMoment(;
                            target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                            w::Union{Nothing, <:AbstractWeights} = nothing,
                            mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return ThirdCentralMoment{typeof(target), typeof(w), typeof(mu)}(target, w, mu)
end
function (r::ThirdCentralMoment)(w::AbstractVector, X::AbstractMatrix,
                                 fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    return sum(val .^ 3) / length(x)
end

export ThirdCentralMoment
