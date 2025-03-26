struct ThirdCentralMoment{T1 <: RiskMeasureSettings,
                          T2 <: Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                          T3 <: Union{Nothing, <:AbstractWeights},
                          T4 <: Union{Nothing, <:AbstractVector{<:Real}}} <:
       TargetNoOptimisationRiskMeasure
    settings::T1
    target::T2
    w::T3
    mu::T4
end
function ThirdCentralMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                            target::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                            w::Union{Nothing, <:AbstractWeights} = nothing,
                            mu::Union{Nothing, <:AbstractVector{<:Real}} = nothing)
    if isa(target, AbstractVector)
        @smart_assert(!isempty(target))
    end
    if isa(mu, AbstractVector)
        @smart_assert(!isempty(mu))
    end
    return ThirdCentralMoment{typeof(settings), typeof(target), typeof(w), typeof(mu)}(settings,
                                                                                       target,
                                                                                       w,
                                                                                       mu)
end
function (r::ThirdCentralMoment)(w::AbstractVector, X::AbstractMatrix,
                                 fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_target_ret_mu(x, w, r)
    val = x .- target
    return sum(val .^ 3) / length(x)
end

export ThirdCentralMoment
